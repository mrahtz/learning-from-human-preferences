import glob
import os.path as osp
import queue
import time

import joblib
import numpy as np
import tensorflow as tf
from numpy.testing import assert_equal

import params as run_params
from baselines import logger
from baselines.a2c.utils import (Scheduler, cat_entropy, discount_with_dones,
                                 find_trainable_variables, make_path, mse)
from baselines.common import explained_variance, set_global_seeds


"""
- states: model state (e.g. LSTM state)
- masks: is only used for stateful models
"""


class Model(object):
    def __init__(self,
                 policy,
                 ob_space,
                 ac_space,
                 nenvs,
                 nsteps,
                 nstack,
                 num_procs,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 lr=7e-4,
                 alpha=0.99,
                 epsilon=1e-5,
                 total_timesteps=int(80e6),
                 lrschedule='linear'):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=num_procs,
            inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(
            sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(
            sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(
            learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            if run_params.params['print_lr']:
                import datetime
                print(str(datetime.datetime.now()), cur_lr)
            td_map = {
                train_model.X: obs,
                A: actions,
                ADV: advs,
                R: rewards,
                LR: cur_lr
            }
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(object):
    def __init__(self,
                 env,
                 model,
                 seg_pipe,
                 log_dir,
                 nsteps=5,
                 nstack=4,
                 gamma=0.99,
                 orig_rewards=False,
                 gen_segs=True):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        # The first stack of 4 frames: the first 3 frames are zeros,
        # with the last frame coming from env.reset().
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.segment = []
        self.seg_pipe = seg_pipe
        if not orig_rewards:
            from reward_predictor import RewardPredictorEnsemble
            self.reward_model = RewardPredictorEnsemble('a2c')
        self.orig_rewards = orig_rewards
        self.gen_segs = gen_segs

        self.n_episodes = [0 for _ in range(nenv)]
        self.true_reward = [0 for _ in range(nenv)]
        self.tr_ops = [tf.Variable(0) for _ in range(nenv)]
        self.summ_ops = [
            tf.summary.scalar('true reward %d' % env_n, self.tr_ops[env_n])
            for env_n in range(nenv)
        ]
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(
            osp.join(log_dir, 'true_reward'), flush_secs=5)
        self.sess.run([op.initializer for op in self.tr_ops])

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def gen_segments(self, mb_obs, mb_dones):
        # Only generate segments from the first environment
        # TODO: is this the right choice...?
        e0_obs = mb_obs[0]
        e0_dones = mb_dones[0]
        assert_equal(e0_obs.shape, (self.nsteps, 84, 84, 4))
        assert_equal(e0_dones.shape, (self.nsteps, ))

        for step in range(self.nsteps):
            if len(self.segment) == 25:
                self.seg_pipe.put(self.segment)
                self.segment = []
                continue
            elif e0_dones[step]:
                # The last segment will probably not be 25 steps long;
                # drop it, so that all segments in the batch are the same
                # length
                self.segment = []
                continue
            self.segment.append(e0_obs[step])

    def run(self):
        nenvs = len(self.env.remotes)
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = \
            [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states,
                                                      self.dones)
            # TODO remove later
            # Offset of 100 so it doesn't interfere with dot finding
            self.obs[:, 0, 0, -1] = 100 + actions[:]
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            # len({obs, rewards, dones}) == nenvs
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
                    if run_params.params['debug']:
                        print("Env %d done" % n)
            # SubprocVecEnv automatically resets when done
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        # i.e. from nsteps, nenvs to nenvs, nsteps
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        # The first entry was just the init state of 'dones' (all False),
        # before we'd actually run any steps, so drop it.
        mb_dones = mb_dones[:, 1:]

        if self.gen_segs:
            self.gen_segments(mb_obs, mb_dones)

        if self.orig_rewards:
            for env_n, obs in enumerate(mb_obs):
                assert_equal(obs.shape, (self.nsteps, 84, 84, 4))
                if run_params.params['debug']:
                    print("Env %d" % env_n)
                    print(mb_actions[env_n])
                    print([
                        self.env.action_meanings[i] for i in mb_actions[env_n]
                    ])
                    print(mb_rewards[env_n])
        else:
            # for the data from each environment
            assert_equal(mb_obs.shape, (nenvs, self.nsteps, 84, 84, 4))
            for env_n, obs in enumerate(mb_obs):
                assert_equal(obs.shape, (self.nsteps, 84, 84, 4))
                if run_params.params['debug']:
                    print("Env %d" % env_n)
                    print(mb_actions[env_n])
                    print([
                        self.env.action_meanings[i] for i in mb_actions[env_n]
                    ])
                rewards = self.reward_model.reward(obs)
                assert_equal(rewards.shape, (self.nsteps, ))
                mb_rewards[env_n] = rewards

        for env_n, (rs, dones) in enumerate(zip(mb_rewards, mb_dones)):
            assert_equal(rs.shape, (self.nsteps, ))
            assert_equal(dones.shape, (self.nsteps, ))
            for step_n in range(self.nsteps):
                self.true_reward[env_n] += rs[step_n]
                if dones[step_n]:
                    if run_params.params['debug']:
                        print("Env %d: episode finished, true reward %d" %
                              (env_n, self.true_reward[env_n]))
                    self.sess.run(self.tr_ops[env_n].assign(
                        self.true_reward[env_n]))
                    summ = self.sess.run(self.summ_ops[env_n])
                    self.writer.add_summary(summ, self.n_episodes[env_n])

                    self.true_reward[env_n] = 0
                    self.n_episodes[env_n] += 1

        mb_obs = mb_obs.reshape(self.batch_ob_shape)

        last_values = self.model.value(self.obs, self.states,
                                       self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(
                zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                # Make sure that the first iteration of the loop inside
                # discount_with_dones picks up 'value' as the initial
                # value of r
                rewards = discount_with_dones(rewards + [value], dones + [0],
                                              self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


def learn(policy,
          env,
          seed,
          seg_pipe,
          go_pipe,
          log_dir,
          nsteps=5,
          nstack=4,
          total_timesteps=int(80e6),
          vf_coef=0.5,
          ent_coef=0.01,
          max_grad_norm=0.5,
          lr=7e-4,
          lrschedule='linear',
          epsilon=1e-5,
          alpha=0.99,
          gamma=0.99,
          log_interval=100,
          load_path=None,
          save_interval=10000,
          orig_rewards=False,
          gen_segs=True):

    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK

    def make_model():
        return Model(
            policy=policy,
            ob_space=ob_space,
            ac_space=ac_space,
            nenvs=nenvs,
            nsteps=nsteps,
            nstack=nstack,
            num_procs=num_procs,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            lr=lr,
            alpha=alpha,
            epsilon=epsilon,
            total_timesteps=total_timesteps,
            lrschedule=lrschedule)

    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))

    print("Initialising model...")
    if load_path is None:
        model = make_model()
    else:
        with open(osp.join(load_path, 'make_model.pkl'), 'rb') as fh:
            make_model = cloudpickle.loads(fh.read())
        model = make_model()
        ckpt_file = glob.glob(osp.join(load_path, 'checkpoint*'))[0]
        print("Loading policy checkpoint from {}...".format(ckpt_file))
        model.load(ckpt_file)

    runner = Runner(
        env,
        model,
        seg_pipe,
        log_dir,
        nsteps=nsteps,
        nstack=nstack,
        gamma=gamma,
        orig_rewards=orig_rewards,
        gen_segs=gen_segs)

    print("Running...")

    # nsteps: e.g. 5
    # nenvs: e.g. 16
    nbatch = nenvs * nsteps
    fps_tstart = time.time()
    fps_nsteps = 0
    train = False
    for update in range(1, total_timesteps // nbatch + 1):
        # Run for nsteps
        obs, states, rewards, masks, actions, values = runner.run()

        # Before we're told to start training, just generate segments
        if not train:
            try:
                go_pipe.get(block=False)
            except queue.Empty:
                if run_params.params['debug']:
                    print("Training signal not yet given; skipping training")
                continue
            else:
                train = True

        policy_loss, value_loss, policy_entropy = model.train(
            obs, states, rewards, masks, actions, values)

        fps_nsteps += nbatch

        if update % log_interval == 0 or update == 1:
            fps = fps_nsteps / (time.time() - fps_tstart)
            fps_nsteps = 0
            fps_tstart = time.time()

            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
        if save_interval and (update % save_interval == 0
                              or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)

    env.close()
