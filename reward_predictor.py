import logging
import os.path as osp
import time

import easy_tf_log
import numpy as np
from numpy.testing import assert_equal
import tensorflow as tf

from reward_predictor_networks import reward_pred_net
from utils import RunningStat, batch_iter


class RewardPredictorEnsemble:
    """
    An ensemble of reward predictors and associated helper functions.

    Modes:
    * Predict reward
      Inputs: one trajectory segment
      Output: predicted reward
    * Predict preference
      Inputs: two trajectory segments
      Output: preference between the two segments
    """

    def __init__(self,
                 cluster_job_name,
                 network_type,
                 lr=1e-4,
                 cluster_dict=None,
                 batchnorm=False,
                 dropout=0.0,
                 n_preds=1,
                 log_dir=None):
        self.n_preds = n_preds
        graph, self.sess = self.init_sess(cluster_dict, cluster_job_name)
        # Why not just use soft device placement? With soft placement,
        # if we have a bug which prevents an operation being placed on the GPU
        # (e.g. we're using uint8s for operations that the GPU can't do),
        # then TensorFlow will be silent and just place the operation on a CPU.
        # Instead, we want to say: if there's a GPU present, definitely try and
        # put things on the GPU. If it fails, tell us!
        if tf.test.gpu_device_name():
            worker_device = "/job:{}/task:0/gpu:0".format(cluster_job_name)
        else:
            worker_device = "/job:{}/task:0".format(cluster_job_name)
        device_setter = tf.train.replica_device_setter(
            cluster=cluster_dict,
            ps_device="/job:ps/task:0",
            worker_device=worker_device)
        self.rps = []
        with graph.as_default():
            for pred_n in range(n_preds):
                with tf.device(device_setter):
                    with tf.variable_scope("pred_{}".format(pred_n)):
                        rp = RewardPredictorNetwork(
                            network_type=network_type,
                            dropout=dropout,
                            batchnorm=batchnorm,
                            lr=lr)
                self.rps.append(rp)
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=1)
            self.summaries = self.add_summary_ops()

        self.reward_ops = [rp.r1 for rp in self.rps]
        self.pred_ops = [rp.pred for rp in self.rps]
        self.train_ops = [rp.train for rp in self.rps]

        self.checkpoint_file = osp.join(log_dir,
                                        'reward_predictor_checkpoints',
                                        'reward_predictor.ckpt')
        self.train_writer = tf.summary.FileWriter(
            osp.join(log_dir, 'reward_predictor', 'train'), flush_secs=5)
        self.test_writer = tf.summary.FileWriter(
            osp.join(log_dir, 'reward_predictor', 'test'), flush_secs=5)

        self.n_steps = 0
        self.r_norm = RunningStat(shape=n_preds)

        misc_logs_dir = osp.join(log_dir, 'reward_pred', 'misc')
        easy_tf_log.set_dir(misc_logs_dir)

    @staticmethod
    def init_sess(cluster_dict, cluster_job_name):
        graph = tf.Graph()
        cluster = tf.train.ClusterSpec(cluster_dict)
        config = tf.ConfigProto(gpu_options={'allow_growth': True})
        server = tf.train.Server(cluster, job_name=cluster_job_name, config=config)
        sess = tf.Session(server.target, graph)
        return graph, sess

    def add_summary_ops(self):
        loss_ops = []
        acc_ops = []
        for rp in self.rps:
            loss_ops.append(rp.loss)
            acc_ops.append(rp.accuracy)
        mean_loss = tf.reduce_mean(loss_ops)
        mean_accuracy = tf.reduce_mean(acc_ops)

        summary_ops = []
        for pred_n in range(self.n_preds):
            name = 'reward_predictor_accuracy_{}'.format(pred_n)
            op = tf.summary.scalar(name, acc_ops[pred_n])
            summary_ops.append(op)
            name = 'reward_predictor_loss_{}'.format(pred_n)
            op = tf.summary.scalar(name, loss_ops[pred_n])
            summary_ops.append(op)
        op = tf.summary.scalar('reward_predictor_accuracy_mean', mean_accuracy)
        summary_ops.append(op)
        op = tf.summary.scalar('reward_predictor_loss_mean', mean_loss)
        summary_ops.append(op)
        summaries = tf.summary.merge(summary_ops)
        return summaries

    def init_network(self, ckpt_path=None):
        if ckpt_path:
            self.saver.restore(self.sess, ckpt_path)
            print("Loaded reward predictor checkpoint from '{}'".format(ckpt_path))
        else:
            self.sess.run(self.init_op)

    def save(self):
        # TODO put back n_steps
        ckpt_name = self.saver.save(self.sess, self.checkpoint_file)
        print("Saved reward predictor checkpoint to '{}'".format(ckpt_name))

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
        assert_equal(len(ensemble_rs), self.n_preds)
        assert_equal(len(ensemble_rs[0]), n_steps)
        logging.debug("Unnormalized rewards:\n%s", ensemble_rs)

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
        logging.debug("Reward mean/stddev:\n%s %s",
                      self.r_norm.mean,
                      self.r_norm.std)
        logging.debug("Normalized rewards:\n%s", ensemble_rs)

        # "...and then averaging the results."
        rs = np.mean(ensemble_rs, axis=0)
        assert_equal(rs.shape, (n_steps, ))
        logging.debug("After ensemble averaging:\n%s", rs)

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

    def train(self, prefs_train, prefs_val, val_interval):
        """
        Train the ensemble for one full epoch
        """
        print("Training/testing with %d/%d preferences" % (len(prefs_train),
                                                           len(prefs_val)))

        start_steps = self.n_steps
        start_time = time.time()
        for _, batch in enumerate(
                batch_iter(prefs_train.prefs, batch_size=32, shuffle=True)):
            # TODO: refactor this so that each can be taken directly from
            # pref_db
            s1s = []
            s2s = []
            prefs = []
            for k1, k2, pref in batch:
                s1s.append(prefs_train.segments[k1])
                s2s.append(prefs_train.segments[k2])
                prefs.append(pref)

            feed_dict = {}
            for rp in self.rps:
                feed_dict[rp.s1] = s1s
                feed_dict[rp.s2] = s2s
                feed_dict[rp.pref] = prefs
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
                prefs = []
                for k1, k2, pref in val_batch:
                    s1s.append(prefs_val.segments[k1])
                    s2s.append(prefs_val.segments[k2])
                    prefs.append(pref)
                feed_dict = {}
                for rp in self.rps:
                    feed_dict[rp.s1] = s1s
                    feed_dict[rp.s2] = s2s
                    feed_dict[rp.pref] = prefs
                    feed_dict[rp.training] = False

                summaries = self.sess.run(self.summaries, feed_dict)
                self.test_writer.add_summary(summaries, self.n_steps)
            end_time = time.time()
            end_steps = self.n_steps

            rate = (end_steps - start_steps) / (end_time - start_time)
            easy_tf_log.logkv('reward_predictor_training_steps_per_second',
                              rate)


class RewardPredictorNetwork:
    """
    Predict the reward that a human would assign to each frame of
    the input trajectory, trained using the human's preferences between
    pairs of trajectories.

    Network inputs:
    - s1/s2     Trajectory pairs
    - pref      Preferences between each pair of trajectories
    Network outputs:
    - r1/r2     Reward predicted for each frame
    - rs1/rs2   Reward summed over all frames for each trajectory
    - pred      Predicted preference
    """

    def __init__(self, network_type, dropout, batchnorm, lr):
        training = tf.placeholder(tf.bool)
        # Each element of the batch is one trajectory segment.
        # (Dimensions are n segments x n frames per segment x ...)
        s1 = tf.placeholder(tf.float32, shape=(None, None, 84, 84, 4))
        s2 = tf.placeholder(tf.float32, shape=(None, None, 84, 84, 4))
        # For each trajectory segment, there is one human judgement.
        pref = tf.placeholder(tf.float32, shape=(None, 2))

        # Concatenate trajectory segments so that the first dimension is just
        # frames (necessary because of conv layer's requirements on input shape)
        s1_unrolled = tf.reshape(s1, [-1, 84, 84, 4])
        s2_unrolled = tf.reshape(s2, [-1, 84, 84, 4])

        # Predict rewards for each frame in the unrolled batch
        _r1 = reward_pred_net(
            network_type,
            s1_unrolled,
            dropout,
            batchnorm,
            reuse=False,
            training=training)
        _r2 = reward_pred_net(
            network_type,
            s2_unrolled,
            dropout,
            batchnorm,
            reuse=True,
            training=training)

        # Shape should be 'unrolled batch size'
        # where 'unrolled batch size' is 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(_r1, 1)
        c2 = tf.assert_rank(_r2, 1)
        with tf.control_dependencies([c1, c2]):
            # Re-roll to 'batch size' x 'n frames per segment'
            __r1 = tf.reshape(_r1, tf.shape(s1)[0:2])
            __r2 = tf.reshape(_r2, tf.shape(s2)[0:2])
        # Shape should be 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(__r1, 2)
        c2 = tf.assert_rank(__r2, 2)
        with tf.control_dependencies([c1, c2]):
            r1 = __r1
            r2 = __r2

        # Sum rewards over all frames in each segment
        _rs1 = tf.reduce_sum(r1, axis=1)
        _rs2 = tf.reduce_sum(r2, axis=1)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_rs1, 1)
        c2 = tf.assert_rank(_rs2, 1)
        with tf.control_dependencies([c1, c2]):
            rs1 = _rs1
            rs2 = _rs2

        # Predict preferences for each segment
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

        preds_correct = tf.equal(tf.argmax(pref, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(preds_correct, tf.float32))

        _loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pref,
                                                           logits=rs)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_loss, 1)
        with tf.control_dependencies([c1]):
            loss = tf.reduce_sum(_loss)

        # Make sure that batch normalization ops are updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # Inputs
        self.training = training
        self.s1 = s1
        self.s2 = s2
        self.pref = pref

        # Outputs
        self.r1 = r1
        self.r2 = r2
        self.rs1 = rs1
        self.rs2 = rs2
        self.pred = pred

        self.accuracy = accuracy
        self.loss = loss
        self.train = train
