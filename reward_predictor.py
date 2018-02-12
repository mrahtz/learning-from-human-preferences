import datetime
import os
import os.path as osp
import pickle
import queue
import time

import numpy as np
import tensorflow as tf
from numpy.testing import assert_equal

import params
from utils import PrefDB, RunningStat

VAL_FRACTION = 0.2


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
        x = tf.layers.batch_normalization(x, training=training)

    if activation == 'relu':
        x = tf.nn.leaky_relu(x, alpha=0.01)
    else:
        raise Exception("Unkown activation for conv_layer", activation)

    return x


def dense_layer(x,
                units,
                name,
                reuse,
                activation=None):
    # TODO: L2 loss

    x = tf.layers.dense(x, units, activation=None, name=name, reuse=reuse)

    if activation == 'relu':
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


def net_handcrafted(s):
    xc, yc = get_position(s)
    a = s[:, 0, 0, -1] - 100

    a1 = tf.cast(tf.equal(a, 1), tf.int64)
    a2 = tf.cast(tf.equal(a, 2), tf.int64)
    a3 = tf.cast(tf.equal(a, 3), tf.int64)
    a4 = tf.cast(tf.equal(a, 4), tf.int64)

    c1 = tf.sign(41 - yc)  # a = 1 (down)
    c2 = tf.sign(41 - xc)  # a = 2 (right)
    c3 = tf.sign(yc - 43)  # a = 3 (up)
    c4 = tf.sign(xc - 43)  # a = 4 (left)

    r = a1 * c1 + \
        a2 * c2 + \
        a3 * c3 + \
        a4 * c4

    c5 = tf.cast(tf.equal(xc, 0), tf.int64)
    c7 = tf.cast(tf.equal(yc, 0), tf.int64)
    # When the dot is right at the edge, it registers as about 82
    c6 = tf.cast(xc >= 82, tf.int64)
    c8 = tf.cast(yc >= 82, tf.int64)

    against_wall = a1 * c8 + \
                   a2 * c6 + \
                   a3 * c7 + \
                   a4 * c5

    r *= 1 - against_wall

    # so that we have something trainable
    r = tf.cast(r, tf.float32) + 0.0 * tf.Variable(0.0)  

    return r


def net_easyfeatures(s):
    # I couldn't get this to work. It might be because I forgot to take care of
    # variable reuse.
    a = s[:, 0, 0, -1] - 100
    xc = tf.cast(tf.reduce_max(tf.argmin(s[..., -1], 1), 1), tf.float32)
    yc = tf.cast(tf.reduce_max(tf.argmin(s[..., -1], 2), 1), tf.float32)

    l1 = tf.Variable(1.0)
    l2 = tf.Variable(1.0)
    l3 = tf.Variable(1.0)
    l4 = tf.Variable(1.0)
    c1 = l1 - xc
    c2 = l2 - yc
    c3 = xc - l3
    c4 = yc - l4

    x = tf.cast(tf.equal(a, 1), tf.float32) * c1 + \
        tf.cast(tf.equal(a, 2), tf.float32) * c2 + \
        tf.cast(tf.equal(a, 3), tf.float32) * c3 + \
        tf.cast(tf.equal(a, 4), tf.float32) * c4

    return x


def net_conv(s, batchnorm, dropout, training, reuse):
    # Page 15:
    # "[The] input is fed through 4 convolutional layers of size 7x7, 5x5, 3x3,
    # and 3x3 with strides 3, 2, 1, 1, each having 16 filters, with leaky ReLU
    # nonlinearities (α = 0.01). This is followed by a fully connected layer of
    # size 64 and then a scalar output. All convolutional layers use batch norm
    # and dropout with α = 0.5 to prevent predictor overfitting"
    x = s

    x = conv_layer(x, 16, 7, 3, batchnorm, training, "c1", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 5, 2, batchnorm, training, "c2", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c3", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c4", reuse, 'relu')

    w, h, c = x.get_shape()[1:]
    x = tf.reshape(x, [-1, int(w * h * c)])

    x = dense_layer(x, 64, "d1", reuse, activation='relu')
    x = dense_layer(x, 1,  "d2", reuse, activation='relu')
    x = x[:, 0]

    return x


def reward_pred_net(s, dropout, batchnorm, reuse, training):
    x = s

    if params.params['network'] == 'handcrafted':
        return net_handcrafted(s)
    elif params.params['network'] == 'easyfeatures':
        return net_easyfeatures(s)
    elif params.params['network'] == 'conv':
        return net_conv(s, batchnorm, dropout, training, reuse)
    else:
        raise Exception("Unknown reward predictor network architecture",
                        params.params['network'])

    return x


def recv_prefs(pref_pipe, pref_db_train, pref_db_val, db_max):
    n_recvd = 0
    while True:
        try:
            s1, s2, mu = pref_pipe.get(timeout=0.1)
            n_recvd += 1
        except queue.Empty:
            break

        if np.random.rand() < VAL_FRACTION:
            print("Sending pref to val")
            pref_db_val.append(s1, s2, mu)
            print("Val len is now {}".format(len(pref_db_val)))
        else:
            print("Sending pref to train")
            pref_db_train.append(s1, s2, mu)
            print("Train len is now {}".format(len(pref_db_train)))

        if len(pref_db_val) > db_max * VAL_FRACTION:
            print("Val database limit reached; dropping first preference")
            pref_db_val.del_first()
        assert len(pref_db_val) <= db_max * VAL_FRACTION
        print("pref_db_val len:", len(pref_db_val))

        if len(pref_db_train) > db_max * (1 - VAL_FRACTION):
            print("Train database limit reached; dropping first preference")
            pref_db_train.del_first()
        assert len(pref_db_train) <= db_max * (1 - VAL_FRACTION)
        print("pref_db_train len:", len(pref_db_train))
    if params.params['debug']:
        print("%d preferences received" % n_recvd)


def save_prefs(log_dir, name, pref_db_train, pref_db_val):
    fname = osp.join(log_dir, "train_{}.pkl".format(name))
    save_pref_db(pref_db_train, fname)
    fname = osp.join(log_dir, "val_{}.pkl".format(name))
    save_pref_db(pref_db_val, fname)


def train_reward_predictor(lr, pref_pipe, go_pipe, load_prefs_dir, log_dir,
                           db_max, rp_ckpt_dir):
    # TODO clean up the checkpoint passing around
    if rp_ckpt_dir is not None:
        load_network = True
    else:
        load_network = False
    reward_model = RewardPredictorEnsemble(
        'train_reward',
        lr,
        log_dir=log_dir,
        load_network=load_network,
        rp_ckpt_dir=rp_ckpt_dir)

    if load_prefs_dir:
        print("Loading preferences...")
        # TODO make this more flexible
        pref_db_train, pref_db_val = load_pref_db(load_prefs_dir)
    else:
        pref_db_val = PrefDB()
        pref_db_train = PrefDB()

    # Page 15: "We collect 500 comparisons from a randomly initialized policy
    # network at the beginning of training"
    if not params.params['skip_prefs']:
        while True:
            if len(pref_db_train) >= params.params['n_initial_prefs']:
                break
            print("Waiting for preferences; %d so far" % len(pref_db_train))
            recv_prefs(pref_pipe, pref_db_train, pref_db_val, db_max)
            if params.params['save_prefs']:
                save_prefs(log_dir, 'initial', pref_db_train, pref_db_val)
            time.sleep(10.0)

    print("Finished accumulating initial preferences at",
          str(datetime.datetime.now()))
    print("({} preferences over {} segments)".format(
        len(pref_db_train.prefs), len(pref_db_train.segments)))

    if params.params['just_prefs']:
        save_prefs(log_dir, 'initial', pref_db_train, pref_db_val)

    if params.params['just_prefs']:
        raise Exception("Preference generation completed")

    if not params.params['no_pretrain']:
        print("Training for %d epochs..." % params.params['n_initial_epochs'])
        # Page 14: "In the Atari domain we also pretrain the reward predictor
        # for 200 epochs before beginning RL training, to reduce the likelihood
        # of irreversibly learning a bad policy based on an untrained
        # predictor."
        for i in range(params.params['n_initial_epochs']):
            print("Epoch %d" % i)
            reward_model.train(pref_db_train, pref_db_val)
            recv_prefs(pref_pipe, pref_db_train, pref_db_val, db_max)
        reward_model.save()
        print("=== Finished initial training at", str(datetime.datetime.now()))

    if params.params['save_pretrain'] or params.params['just_pretrain']:
        fname = osp.join(log_dir, "train_postpretrain.pkl")
        save_pref_db(pref_db_train, fname)
        fname = osp.join(log_dir, "val_postpretrain.pkl")
        save_pref_db(pref_db_val, fname)
    if params.params['just_pretrain']:
        raise Exception("Pretraining completed")

    print("=== Starting RL training at", str(datetime.datetime.now()))
    # Start RL training
    go_pipe.put(True)

    step = 0
    prev_save_step = None
    while True:
        reward_model.train(pref_db_train, pref_db_val)
        recv_prefs(pref_pipe, pref_db_train, pref_db_val, db_max)

        if params.params['save_prefs'] and \
                step % params.params['save_freq'] == 0:
            print("=== Saving preferences...")
            fname = osp.join(log_dir, "train_%d.pkl" % step)
            save_pref_db(pref_db_train, fname)
            fname = osp.join(log_dir, "val_%d.pkl" % step)
            save_pref_db(pref_db_val, fname)
            if prev_save_step is not None:
                os.remove(osp.join(log_dir, "train_%d.pkl" % prev_save_step))
                os.remove(osp.join(log_dir, "val_%d.pkl" % prev_save_step))
            prev_save_step = step

        step += 1


def save_pref_db(pref_db, fname):
    with open(fname, 'wb') as pkl_file:
        pickle.dump(pref_db, pkl_file)


def load_pref_db(pref_dir):
    train_fname = osp.join(pref_dir, 'train_postpretrain.pkl')
    if not os.path.isfile(train_fname):
        train_fname = osp.join(pref_dir, 'train_initial.pkl')
    with open(train_fname, 'rb') as pkl_file:
        print("Loading training preferences from '{}'".format(train_fname))
        pref_db_train = pickle.load(pkl_file)

    val_fname = osp.join(pref_dir, 'val_postpretrain.pkl')
    if not os.path.isfile(val_fname):
        val_fname = osp.join(pref_dir, 'val_initial.pkl')
    with open(val_fname, 'rb') as pkl_file:
        print("Loading validation preferences from '{}'".format(val_fname))
        pref_db_val = pickle.load(pkl_file)

    return pref_db_train, pref_db_val


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
                 lr=1e-4,
                 log_dir='/tmp',
                 cluster_dict=None,
                 n_preds=1,
                 load_network=False,
                 rp_ckpt_dir=None,
                 dropout=0.5):
        rps = []
        reward_ops = []
        pred_ops = []
        train_ops = []
        loss_ops = []
        acc_ops = []
        graph = tf.Graph()

        if cluster_dict is None:
            cluster_dict = {
                'a2c': ['localhost:2200'],
                'pref_interface': ['localhost:2201'],
                'train_reward': ['localhost:2202'],
                'ps': ['localhost:2203']
            }
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
                        batchnorm = params.params['batchnorm']
                        rp = RewardPredictor(
                            dropout=dropout, batchnorm=batchnorm, lr=lr)
                reward_ops.append(rp.r1)
                pred_ops.append(rp.pred)
                train_ops.append(rp.train)
                loss_ops.append(rp.loss)
                acc_ops.append(rp.accuracy)
                rps.append(rp)

            self.mean_loss = tf.reduce_mean(loss_ops)
            # TODO: this probably isn't representative of the accuracy of the
            # ensemble as a whole; the ensemble's predictions are arrived at
            # by voting, whereas this just takes the mean of the accuracies
            # of each ensemble member individually
            self.mean_accuracy = tf.reduce_mean(acc_ops)

            self.saver = tf.train.Saver(max_to_keep=1)
            self.checkpoint_file = osp.join(log_dir, 'checkpoints',
                                            'reward_network.ckpt')

            # Only the reward predictor training process should initialize the
            # network
            if name != 'train_reward':
                while len(sess.run(tf.report_uninitialized_variables())) > 0:
                    print("%s waiting for variable initialization..." % name)
                    time.sleep(1.0)
            else:
                if load_network:
                    # TODO fix
                    ckpt_f = rp_ckpt_dir
                    print("Loading reward predictor checkpoint from", ckpt_f)
                    self.saver.restore(sess, ckpt_f)
                    print("Reard predictor checkpoint loaded!")
                else:
                    sess.run(tf.global_variables_initializer())

        if name == 'ps':
            server.join()

        self.acc_summ = tf.summary.scalar('reward predictor accuracy',
                                          self.mean_accuracy)
        self.loss_summ = tf.summary.scalar('reward predictor loss',
                                           self.mean_loss)
        self.summaries = tf.summary.merge([self.acc_summ, self.loss_summ])

        self.train_writer = tf.summary.FileWriter(
            osp.join(log_dir, 'reward_pred', 'train'), flush_secs=5)
        self.test_writer = tf.summary.FileWriter(
            osp.join(log_dir, 'reward_pred', 'test'), flush_secs=5)

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

    def raw_rewards(self, obs):
        """
        Return reward for each frame, returned directly from each member
        of the ensemble (before any normalization or averaging has taken place)
        """
        n_steps = obs.shape[0]
        assert_equal(obs.shape, (n_steps, 84, 84, 4))

        feed_dict = {}
        for rp in self.rps:
            feed_dict[rp.training] = False
            feed_dict[rp.s1] = [obs]
        # This will return nested lists of sizes n_preds x 1 x nsteps
        # (x 1 because of the batch size of 1)
        rs = self.sess.run(self.reward_ops, feed_dict)
        # Get rid of the extra x 1 dimension
        for i in range(len(rs)):
            rs[i] = rs[i][0]

        # Final shape should be 'n_preds x n_steps'
        assert_equal(len(rs), len(self.rps))
        assert_equal(len(rs[0]), n_steps)

        return rs

    def reward_unnormalized(self, obs):
        """
        Return reward for each frame, averaged over all ensemble members.
        """
        n_steps = obs.shape[0]
        assert_equal(obs.shape, (n_steps, 84, 84, 4))
        rs = self.raw_rewards(obs)
        # Shape should be 'n_preds x n_steps'
        assert_equal(len(rs), len(self.rps))
        assert_equal(len(rs[0]), n_steps)

        rs = np.mean(rs, axis=0)
        assert_equal(rs.shape, (n_steps, ))

        return rs

    def reward(self, obs):
        """
        Return reward for each frame, normalized to have zero mean and constant
        standard deviation separately for each member of the ensemble, then
        averaged across all members of the ensemble.
        """
        n_steps = obs.shape[0]
        assert_equal(obs.shape, (n_steps, 84, 84, 4))

        ensemble_rs = self.raw_rewards(obs)
        # Shape should be 'n_preds x n_steps'
        assert_equal(len(ensemble_rs), len(self.rps))
        assert_equal(len(ensemble_rs[0]), n_steps)

        if params.params['debug']:
            print("Raw rewards:", ensemble_rs)

        # TODO: I'm assuming that normalization is only applied
        # to the rewards fed to the policy network
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

        # ensemble_rs is n_preds x n_steps
        ensemble_rs = np.array(ensemble_rs).transpose()
        # now n_steps x n_preds

        # TODO re-enable this!
        """
        for ensemble_rs_step in ensemble_rs:
            self.r_norm.push(ensemble_rs_step)
        ensemble_rs -= self.r_norm.mean
        ensemble_rs /= (self.r_norm.std + 1e-12)
        ensemble_rs *= 0.05
        """

        ensemble_rs = ensemble_rs.transpose()
        # now n_preds x n_steps again

        if params.params['debug']:
            print("Reward mean/stddev:", self.r_norm.mean, self.r_norm.std)
            print("Ensemble rewards post-normalisation:", ensemble_rs)

        assert_equal(len(ensemble_rs), len(self.rps))
        assert_equal(len(ensemble_rs[0]), n_steps)

        # "...and then averaging the results."
        rs = np.mean(ensemble_rs, axis=0)
        assert_equal(rs.shape, (n_steps, ))
        if params.params['debug']:
            print("Sending back rewards", rs)

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
        ckpt_name = self.saver.save(self.sess, self.checkpoint_file,
                                    self.n_steps)
        return ckpt_name

    def train(self, prefs_train, prefs_val, test_interval=10):
        """
        Train the ensemble for one full epoch
        """
        print("Training/testing with %d/%d preferences" % (len(prefs_train),
                                                           len(prefs_val)))

        for batch_n, batch in enumerate(
                batch_iter(prefs_train.prefs, batch_size=32, shuffle=True)):
            n_prefs = len(batch)
            print("Batch %d: %d preferences" % (batch_n, n_prefs))
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

            if self.n_steps and self.n_steps % test_interval == 0:
                if len(prefs_val) <= 32:
                    val_batch = prefs_val.prefs
                else:
                    idxs = np.random.choice(
                        len(prefs_val.prefs), 32, replace=True)
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

                acc_summ, accuracies, losses = self.sess.run(
                    [self.acc_summ, self.acc_ops, self.loss_ops], feed_dict)
                if params.params['debug']:
                    print("Accuracies:", accuracies)
                    print("Losses:", losses)
                self.test_writer.add_summary(acc_summ, self.n_steps)

            print("Trained %d steps" % self.n_steps)

            self.n_steps += 1

            if self.n_steps % params.params['ckpt_freq'] == 0:
                print("=== Saving reward predictor checkpoint...")
                self.save()


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

    def __init__(self, dropout, batchnorm, lr):
        training = tf.placeholder(tf.bool)

        # Each element of the batch is one trajectory segment.
        # (Dimensions are n segments x n frames per segment x ...)
        s1 = tf.placeholder(tf.float32, shape=(None, None, 84, 84, 4))
        s2 = tf.placeholder(tf.float32, shape=(None, None, 84, 84, 4))
        # For each trajectory segment, there is one human judgement.
        mu = tf.placeholder(tf.float32, shape=(None, 2))

        # Concatenate trajectory segments, so that the first dimension
        # is just frames
        s1_unrolled = tf.reshape(s1, [-1, 84, 84, 4], name='a')
        s2_unrolled = tf.reshape(s2, [-1, 84, 84, 4], name='b')

        _r1 = reward_pred_net(
            s1_unrolled, dropout, batchnorm, reuse=None, training=training)
        _r2 = reward_pred_net(
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

        # Prevent tf.nn.softmax_cross_entropy_with_logits_v2 from propagating
        # gradients into labels
        _mu = tf.stop_gradient(mu)
        _loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=_mu, logits=rs)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_loss, 1)
        with tf.control_dependencies([c1]):
            loss = tf.reduce_sum(_loss)

        # Make sure that batch normalization ops are updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # TODO
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
