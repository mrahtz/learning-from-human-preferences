import datetime
import os
import os.path as osp
import pickle
import queue
import time

import numpy as np
from numpy.testing import assert_equal
import tensorflow as tf

import params
from utils import RunningStat
import logging




def batch_iter(data, batch_size, shuffle=False):
    idxs = list(range(len(data)))
    if shuffle:
        # Yes, this really does shuffle in-place
        np.random.shuffle(idxs)

    start_idx = 0
    end_idx = 0
    while end_idx < len(data):
        end_idx = start_idx + batch_size
        if end_idx > len(data):
            end_idx = len(data)

        batch_idxs = idxs[start_idx:end_idx]
        batch = []
        for idx in batch_idxs:
            batch.append(data[idx])

        yield batch
        start_idx += batch_size


def conv_layer(x, filters, kernel_size, strides, batchnorm, training, name,
               reuse, activation='relu'):
    # TODO: L2 loss
    x = tf.layers.conv2d(
        x,
        filters,
        kernel_size,
        strides,
        activation=None,
        name=name,
        reuse=reuse)

    if batchnorm:
        batchnorm_name = name + "_batchnorm"
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=batchnorm_name)

    if activation == 'relu':
        x = tf.nn.leaky_relu(x, alpha=0.01)
    else:
        raise Exception("Unknown activation for conv_layer", activation)

    return x


def dense_layer(x,
                units,
                name,
                reuse,
                activation=None):
    # TODO: L2 loss

    x = tf.layers.dense(x, units, activation=None, name=name, reuse=reuse)

    if activation is None:
        pass
    elif activation == 'relu':
        x = tf.nn.leaky_relu(x, alpha=0.01)
    else:
        raise Exception("Unknown activation for dense_layer", activation)

    return x


def get_position(s):
    # s is (?, 84, 84, 4)
    s = s[..., -1]  # select last frame; now (?, 84, 84)

    x = tf.reduce_sum(s, axis=1)  # now (?, 84)
    x = tf.argmax(x, axis=1)

    y = tf.reduce_sum(s, axis=2)
    y = tf.argmax(y, axis=1)

    return x, y


def net_easyfeatures(s, reuse):
    a = s[:, 0, 0, -1] - 100
    a = tf.cast(a, tf.float32) / 4.0

    xc, yc = get_position(s)
    xc = tf.cast(xc, tf.float32) / 83.0
    yc = tf.cast(yc, tf.float32) / 83.0

    features = [a, xc, yc]
    x = tf.stack(features, axis=1)

    x = dense_layer(x, 64, "d1", reuse, activation='relu')
    x = dense_layer(x, 64, "d2", reuse, activation='relu')
    x = dense_layer(x, 64, "d3", reuse, activation='relu')
    x = dense_layer(x, 1, "d4", reuse, activation=None)
    x = x[:, 0]

    return x


def net_conv(s, batchnorm, dropout, training, reuse):
    # Page 15:
    # "[The] input is fed through 4 convolutional layers of size 7x7, 5x5, 3x3,
    # and 3x3 with strides 3, 2, 1, 1, each having 16 filters, with leaky ReLU
    # nonlinearities (α = 0.01). This is followed by a fully connected layer of
    # size 64 and then a scalar output. All convolutional layers use batch norm
    # and dropout with α = 0.5 to prevent predictor overfitting"
    x = tf.cast(s, tf.float32) / 255.0

    x = conv_layer(x, 16, 7, 3, batchnorm, training, "c1", reuse, 'relu')
    # NB specifying seed is important because both legs of the network should dropout
    # in the same way.
    # TODO: this still isn't completely right; we should set noise_shape for same dropout on all steps
    x = tf.layers.dropout(x, dropout, training=training, seed=0)
    x = conv_layer(x, 16, 5, 2, batchnorm, training, "c2", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training, seed=1)
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c3", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training, seed=2)
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c4", reuse, 'relu')

    w, h, c = x.get_shape()[1:]
    x = tf.reshape(x, [-1, int(w * h * c)])

    x = dense_layer(x, 64, "d1", reuse, activation='relu')
    x = dense_layer(x, 1, "d2", reuse, activation=None)
    x = x[:, 0]

    return x


def reward_pred_net(network, s, dropout, batchnorm, reuse, training):
    if network == 'moving_dot_features':
        return net_easyfeatures(s, reuse)
    elif network == 'cnn':
        return net_conv(s, batchnorm, dropout, training, reuse)
    else:
        raise Exception("Unknown reward predictor network architecture",
                        network)


class RewardPredictorEnsemble:
    """
    Modes:
    * Predict reward
      Inputs: one trajectory segment
      Output: predicted reward
    * Predict preference
      Inputs: two trajectory segments
      Output: preference between the two segments
    """

    def __init__(self,
                 name,
                 network,
                 lr=1e-4,
                 cluster_dict=None,
                 batchnorm=False,
                 dropout=0.0,
                 n_preds=1,
                 log_dir=None):
        rps = []
        reward_ops = []
        pred_ops = []
        train_ops = []
        loss_ops = []
        acc_ops = []
        graph = tf.Graph()

        cluster = tf.train.ClusterSpec(cluster_dict)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        server = tf.train.Server(cluster, job_name=name, config=config)
        sess = tf.Session(server.target, graph)

        device_setter = tf.train.replica_device_setter(
            cluster=cluster_dict,
            ps_device="/job:ps/task:0",
            worker_device="/job:{}/task:0".format(name))

        with graph.as_default():
            for i in range(n_preds):
                with tf.device(device_setter):
                    with tf.variable_scope("pred_%d" % i):
                        rp = RewardPredictor(
                            network=network, dropout=dropout, batchnorm=batchnorm, lr=lr)
                reward_ops.append(rp.r1)
                pred_ops.append(rp.pred)
                train_ops.append(rp.train)
                loss_ops.append(rp.loss)
                acc_ops.append(rp.accuracy)
                rps.append(rp)
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=1)
            mean_loss = tf.reduce_mean(loss_ops)
            mean_accuracy = tf.reduce_mean(acc_ops)

        summary_ops = []
        op = tf.summary.scalar('reward_predictor_accuracy_mean', mean_accuracy)
        summary_ops.append(op)
        op = tf.summary.scalar('reward_predictor_loss_mean', mean_loss)
        summary_ops.append(op)
        for i in range(n_preds):
            name = 'reward_predictor_accuracy_{}'.format(i)
            op = tf.summary.scalar(name, acc_ops[i])
            summary_ops.append(op)
            name = 'reward_predictor_loss_{}'.format(i)
            op = tf.summary.scalar(name, loss_ops[i])
            summary_ops.append(op)
        self.summaries = tf.summary.merge(summary_ops)

        self.sess = sess
        self.rps = rps
        self.reward_ops = reward_ops
        self.train_ops = train_ops
        self.loss_ops = loss_ops
        self.acc_ops = acc_ops
        self.pred_ops = pred_ops
        self.n_preds = n_preds
        self.n_steps = 0
        self.r_norm = RunningStat(shape=n_preds)
        self.name = name
        self.server = server

        self.checkpoint_file = osp.join(log_dir,
                                        'reward_predictor_checkpoints',
                                        'reward_predictor.ckpt')
        self.train_writer = tf.summary.FileWriter(
            osp.join(log_dir, 'reward_pred', 'train'), flush_secs=5)
        self.test_writer = tf.summary.FileWriter(
            osp.join(log_dir, 'reward_pred', 'test'), flush_secs=5)

    def wait_for_init(self):
        while self.sess.run(tf.report_uninitialized_variables()).any():
            print("{} waiting for variable initialization...".format(self.name))
            time.sleep(1.0)

    def init_network(self, ckpt_path=None):
        if ckpt_path:
            self.saver.restore(self.sess, ckpt_path)
            print("Loaded reward predictor checkpoint from", ckpt_path)
        else:
            self.sess.run(self.init_op)


    def raw_rewards(self, obs):
        """
        Return (unnormalized) reward for each frame from each member of the
        ensemble.
        """
        n_steps = obs.shape[0]
        assert_equal(obs.shape, (n_steps, 84, 84, 4))

        feed_dict = {}
        for rp in self.rps:
            feed_dict[rp.training] = False
            # reward_ops corresponds to the rewards calculated from the
            # s1 input
            feed_dict[rp.s1] = [obs]
        # This will return nested lists of sizes n_preds x 1 x nsteps
        # (x 1 because of the batch size of 1)
        rs = self.sess.run(self.reward_ops, feed_dict)
        rs = np.array(rs)
        # Get rid of the extra x 1 dimension
        rs = rs[:, 0, :]
        assert_equal(rs.shape, (self.n_preds, n_steps))

        return rs

    def reward(self, obs):
        """
        Return (normalized) reward for each frame.

        (Normalization involves normalizing the rewards from each member of the
        ensemble separately (zero mean and constant standard deviation), then
        averaging the resulting rewards across all ensemble members.)
        """
        n_steps = obs.shape[0]
        assert_equal(obs.shape, (n_steps, 84, 84, 4))

        # Get unnormalized rewards

        ensemble_rs = self.raw_rewards(obs)
        # Shape should be 'n_preds x n_steps'
        assert_equal(len(ensemble_rs), len(self.rps))
        assert_equal(len(ensemble_rs[0]), n_steps)
        logging.debug("Unnormalized rewards:", ensemble_rs)

        # Normalize rewards

        # Note that we don't just implement reward normalization in the network
        # graph itself because:
        # * It's simpler not to do it in TensorFlow
        # * Preference calculation doesn't need normalized rewards. Only
        #   rewards sent to the the RL algorithm need to be normalized.
        #   So we can save on computation.

        # Page 4:
        # "We normalized the rewards produced by r^ to have zero mean and
        #  constant standard deviation."
        # Page 15:
        # "Since the reward predictor is ultimately used to compare two sums
        #  over timesteps, its scale is arbitrary, and we normalize it to have
        #  a standard deviation of 0.05"
        # Page 5:
        # "The estimate r^ is defined by independently normalizing each of
        #  these predictors..."

        # We want to keep track of running mean/stddev for each member of the
        # ensemble separately, so we have to be a little careful here.
        assert_equal(ensemble_rs.shape, (self.n_preds, n_steps))
        ensemble_rs = ensemble_rs.transpose()
        assert_equal(ensemble_rs.shape, (n_steps, self.n_preds))
        for ensemble_rs_step in ensemble_rs:
            self.r_norm.push(ensemble_rs_step)
        ensemble_rs -= self.r_norm.mean
        ensemble_rs /= (self.r_norm.std + 1e-12)
        ensemble_rs *= 0.05
        ensemble_rs = ensemble_rs.transpose()
        assert_equal(ensemble_rs.shape, (self.n_preds, n_steps))
        logging.debug("Reward mean/stddev:", self.r_norm.mean, self.r_norm.std)
        logging.debug("Normalized rewards:", ensemble_rs)

        # "...and then averaging the results."
        rs = np.mean(ensemble_rs, axis=0)
        assert_equal(rs.shape, (n_steps, ))
        logging.debug("After ensemble averaging:", rs)

        return rs

    def preferences(self, s1s, s2s):
        """
        Predict probability of human preferring one segment over another
        for each segment in the supplied segment pairs.
        """
        feed_dict = {}
        for rp in self.rps:
            feed_dict[rp.s1] = s1s
            feed_dict[rp.s2] = s2s
            feed_dict[rp.training] = False
        preds = self.sess.run(self.pred_ops, feed_dict)
        return preds

    def save(self):
        # TODO put back n_steps
        ckpt_name = self.saver.save(self.sess, self.checkpoint_file)
        print("Saved reward predictor checkpoint to '{}'".format(ckpt_name))

    def train(self, prefs_train, prefs_val, val_interval):
        """
        Train the ensemble for one full epoch
        """
        print("Training/testing with %d/%d preferences" % (len(prefs_train),
                                                           len(prefs_val)))

        for batch_n, batch in enumerate(
                batch_iter(prefs_train.prefs, batch_size=32, shuffle=True)):
            # TODO: refactor this so that each can be taken directly from
            # pref_db
            s1s = []
            s2s = []
            mus = []
            for k1, k2, mu in batch:
                s1s.append(prefs_train.segments[k1])
                s2s.append(prefs_train.segments[k2])
                mus.append(mu)

            feed_dict = {}
            for rp in self.rps:
                feed_dict[rp.s1] = s1s
                feed_dict[rp.s2] = s2s
                feed_dict[rp.mu] = mus
                feed_dict[rp.training] = True
            summaries, _ = self.sess.run([self.summaries, self.train_ops],
                                         feed_dict)
            self.train_writer.add_summary(summaries, self.n_steps)
            self.n_steps += 1
            print("Trained reward predictor for %d steps" % self.n_steps)

            if self.n_steps and self.n_steps % val_interval == 0:
                if len(prefs_val) <= 32:
                    val_batch = prefs_val.prefs
                else:
                    idxs = np.random.choice(
                        len(prefs_val.prefs), 32, replace=False)
                    val_batch = []
                    for idx in idxs:
                        val_batch.append(prefs_val.prefs[idx])
                s1s = []
                s2s = []
                mus = []
                for k1, k2, mu in val_batch:
                    s1s.append(prefs_val.segments[k1])
                    s2s.append(prefs_val.segments[k2])
                    mus.append(mu)
                feed_dict = {}
                for rp in self.rps:
                    feed_dict[rp.s1] = s1s
                    feed_dict[rp.s2] = s2s
                    feed_dict[rp.mu] = mus
                    feed_dict[rp.training] = False

                summaries = self.sess.run(self.summaries, feed_dict)
                self.test_writer.add_summary(summaries, self.n_steps)


class RewardPredictor:
    """
    Inputs:
    - s1/s2     Two batches of trajectories
    - mu        One batch of preferences between each pair of trajectories
    Outputs:
    - r1/r2     Reward predicted for each frame
    - rs1/rs2   Reward summed over all frames for each trajectory
    - pred      Predicted preference
    - loss      Loss summed over the whole batch
    """

    def __init__(self, network, dropout, batchnorm, lr):
        training = tf.placeholder(tf.bool)

        # Each element of the batch is one trajectory segment.
        # (Dimensions are n segments x n frames per segment x ...)
        s1 = tf.placeholder(tf.uint8, shape=(None, None, 84, 84, 4))
        s2 = tf.placeholder(tf.uint8, shape=(None, None, 84, 84, 4))
        # For each trajectory segment, there is one human judgement.
        mu = tf.placeholder(tf.float32, shape=(None, 2))

        # Concatenate trajectory segments, so that the first dimension
        # is just frames
        s1_unrolled = tf.reshape(s1, [-1, 84, 84, 4], name='a')
        s2_unrolled = tf.reshape(s2, [-1, 84, 84, 4], name='b')

        _r1 = reward_pred_net(network,
            s1_unrolled, dropout, batchnorm, reuse=None, training=training)
        _r2 = reward_pred_net(network,
            s2_unrolled, dropout, batchnorm, reuse=True, training=training)

        # Shape should be 'unrolled batch size'
        # where 'unrolled batch size' is 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(_r1, 1)
        c2 = tf.assert_rank(_r2, 1)
        with tf.control_dependencies([c1, c2]):
            __r1 = tf.reshape(_r1, tf.shape(s1)[0:2], name='c')
            __r2 = tf.reshape(_r2, tf.shape(s2)[0:2], name='d')

        # Shape should be 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(__r1, 2)
        c2 = tf.assert_rank(__r2, 2)
        with tf.control_dependencies([c1, c2]):
            r1 = __r1
            r2 = __r2

        _rs1 = tf.reduce_sum(r1, axis=1)
        _rs2 = tf.reduce_sum(r2, axis=1)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_rs1, 1)
        c2 = tf.assert_rank(_rs2, 1)
        with tf.control_dependencies([c1, c2]):
            rs1 = _rs1
            rs2 = _rs2

        _rs = tf.stack([rs1, rs2], axis=1)

        # Shape should be 'batch size' x 2
        c1 = tf.assert_rank(_rs, 2)
        with tf.control_dependencies([c1]):
            rs = _rs

        _pred = tf.nn.softmax(rs)

        # Shape should be 'batch_size' x 2
        c1 = tf.assert_rank(_pred, 2)
        with tf.control_dependencies([c1]):
            pred = _pred

        preds_correct = tf.equal(tf.argmax(mu, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(preds_correct, tf.float32))

        _loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=mu, logits=rs)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_loss, 1)
        with tf.control_dependencies([c1]):
            loss = tf.reduce_sum(_loss)

        # Make sure that batch normalization ops are updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # TODO (L2 loss)
        """
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.add_n(reg_losses)
        loss += reg_loss
        """

        self.s1 = s1
        self.s2 = s2
        self.mu = mu
        self.r1 = r1
        self.r2 = r2
        self.rs1 = rs1
        self.rs2 = rs2
        self.rs = rs
        self.mu = mu
        self.pred = pred
        self.training = training
        self._loss = _loss
        self.loss = loss
        self.accuracy = accuracy
        self.train = train
