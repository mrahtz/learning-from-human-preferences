#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
import logging
import os
import os.path as osp
import subprocess
import sys
import time
from multiprocessing import Process, Queue

import gym
import gym_moving_dot  # noqa: F401 (imported but unused)
import easy_tf_log
import params

from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble, train_reward_predictor
from enduro_wrapper import EnduroWrapper
from utils import vid_proc, get_port_range

sys.path.insert(0, 'baselines')
from baselines import bench, logger  # noqa: E402 (import not at top of file)
from baselines.a2c.a2c import learn  # noqa: E402
from baselines.a2c.policies import MlpPolicy, CnnPolicy  # noqa: E402
from baselines.a2c.utils import Scheduler  # noqa: E402
from baselines.common import set_global_seeds  # noqa: E402
from baselines.common.atari_wrappers import wrap_deepmind_nomax  # noqa: E402
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv  # noqa: E402, E501

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def configure_logger(log_dir):
    baselines_dir = osp.join(log_dir, 'baselines')
    os.makedirs(baselines_dir)

    json_file = open(osp.join(baselines_dir, 'progress.json'), 'wt')
    json = logger.JSONOutputFormat(json_file)

    tb = logger.TensorBoardOutputFormat(baselines_dir)

    formats = [json, tb]
    logger.Logger.CURRENT = logger.Logger(
        dir=baselines_dir, output_formats=formats)


def train(env_id, num_timesteps, seed, lr_scheduler, rp_lr, num_cpu,
          rp_ckpt_path, load_prefs_dir, headless, log_dir, ent_coef, db_max,
          segs_max, log_interval, policy_ckpt_dir, policy_ckpt_interval):
    configure_logger(log_dir)

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            if params.params['env'] == 'EnduroNoFrameskip-v4':
                env = EnduroWrapper(env)
            env = bench.Monitor(
                env,
                logger.get_dir()
                and osp.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind_nomax(env)

        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if params.params['policy'] == 'mlp':
        policy_fn = MlpPolicy
    elif params.params['policy'] == 'cnn':
        policy_fn = CnnPolicy
    else:
        raise Exception("Unknown policy {}".format(params.params['policy']))

    seg_pipe = Queue()
    pref_pipe = Queue()
    go_pipe = Queue(maxsize=1)

    def episode_vid_proc():
        vid_proc(
            episode_vid_queue,
            playback_speed=2,
            zoom_factor=2,
            mode='play_through')

    if params.params['render_episodes']:
        episode_vid_queue = Queue()
        Process(
            target=episode_vid_proc,
            daemon=True).start()
    else:
        episode_vid_queue = None

    parts_to_run = ['a2c', 'pref_interface', 'train_reward']

    if params.params['no_gather_prefs'] or params.params['orig_rewards']:
        parts_to_run.remove('pref_interface')
    if params.params['orig_rewards']:
        parts_to_run.remove('train_reward')
    if params.params['no_a2c']:
        parts_to_run.remove('a2c')

    ports = get_port_range(
        start_port=2200,
        n_ports=len(parts_to_run) + 1,
        random_stagger=True)
    cluster_dict = {'ps': ['localhost:{}'.format(ports[0])]}
    for part, port in zip(parts_to_run, ports[1:]):
        cluster_dict[part] = ['localhost:{}'.format(port)]

    def ps():
        RewardPredictorEnsemble(name='ps', cluster_dict=cluster_dict)

    def a2c():
        if params.params['orig_rewards']:
            reward_predictor = None
        else:
            reward_predictor = RewardPredictorEnsemble(
                name='a2c',
                cluster_dict=cluster_dict)
        learn(
            policy=policy_fn,
            env=env,
            seed=seed,
            seg_pipe=seg_pipe,
            go_pipe=go_pipe,
            log_dir=log_dir,
            lr_scheduler=lr_scheduler,
            total_timesteps=num_timesteps,
            log_interval=log_interval,
            load_path=policy_ckpt_dir,
            reward_predictor=reward_predictor,
            episode_vid_queue=episode_vid_queue,
            ent_coef=ent_coef,
            save_interval=policy_ckpt_interval)

    def trp():
        reward_predictor = RewardPredictorEnsemble(
            name='train_reward',
            cluster_dict=cluster_dict,
            log_dir=log_dir,
            ckpt_path=rp_ckpt_path,
            lr=rp_lr)
        train_reward_predictor(
            reward_predictor=reward_predictor,
            pref_pipe=pref_pipe,
            go_pipe=go_pipe,
            load_prefs_dir=load_prefs_dir,
            log_dir=log_dir,
            db_max=db_max)

    def pi_procf(pi):
        # We have to give PrefInterface the reward predictor /after/ init,
        # because init needs to spawn a GUI process (using fork()), and the
        # Objective C APIs (invoked when dealing with GUI stuff) aren't
        # happy if running in a process forked from a multithreaded parent.
        reward_predictor = RewardPredictorEnsemble(
            name='pref_interface',
            cluster_dict=cluster_dict)
        pi.init_reward_predictor(reward_predictor)
        pi.run(seg_pipe, pref_pipe, segs_max)

    ps_proc = Process(target=ps, daemon=True)
    ps_proc.start()

    train_proc = None
    if 'train_reward' in parts_to_run:
        train_proc = Process(target=trp, daemon=True)
        train_proc.start()

    a2c_proc = None
    if 'a2c' in parts_to_run:
        a2c_proc = Process(target=a2c, daemon=True)
        a2c_proc.start()

    if 'pref_interface' in parts_to_run:
        synthetic_prefs = headless
        pi = PrefInterface(headless, synthetic_prefs)
        pi_proc = Process(target=pi_procf, daemon=True, args=(pi, ))
        pi_proc.start()

    if 'train_reward' not in parts_to_run:
        go_pipe.put(True)

    if params.params['just_prefs'] or params.params['just_pretrain']:
        train_proc.join()
    elif a2c_proc:
        a2c_proc.join()
    else:
        raise Exception("Error: no parts to wait for?")

    env.close()  # Kill the SubprocVecEnv processes
    if 'train_proc' in parts_to_run:
        train_proc.terminate()
    if 'pref_interface' in parts_to_run:
        pi_proc.terminate()
    ps_proc.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env', help='environment ID', default='EnduroNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--rp_lr', type=float, default=2e-4)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument("--lr_zero_million_timesteps",
                        type=float, default=None,
                        help='If set, decay learning rate linearly, reaching '
                        ' zero at this many timesteps')
    parser.add_argument('--million_timesteps',
                        type=float, default=10.,
                        help='How many million timesteps to train for. '
                             '(The number of frames trained for is this '
                             'multiplied by 4 due to frameskip.)')
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--load_prefs_dir')
    parser.add_argument('--headless', action='store_true')
    seconds_since_epoch = str(int(time.time()))
    parser.add_argument('--run_name', default=seconds_since_epoch)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--db_max', type=int, default=3000)
    parser.add_argument('--segs_max', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--just_pretrain', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--save_prefs', action='store_true')
    parser.add_argument('--network', default='conv')
    parser.add_argument('--just_prefs', action='store_true')
    parser.add_argument('--save_pretrain', action='store_true')
    parser.add_argument('--print_lr', action='store_true')
    parser.add_argument('--n_initial_epochs', type=int, default=200)
    parser.add_argument('--log_dir')
    parser.add_argument('--orig_rewards', action='store_true')
    parser.add_argument('--skip_prefs', action='store_true')
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_preds', type=int, default=1)
    parser.add_argument('--save_initial_prefs', action='store_true')
    parser.add_argument('--random_queries', action='store_true')
    parser.add_argument('--no_gather_prefs', action='store_true')
    parser.add_argument('--no_a2c', action='store_true')
    parser.add_argument('--render_episodes', action='store_true')

    # Reward predictor options
    parser.add_argument('--n_initial_prefs', type=int, default=500,
                        help='How many preferences to collect from a random '
                             'policy before starting reward predictor '
                             'training')
    parser.add_argument('--load_reward_predictor_ckpt',
                        help='File to load reward predictor checkpoint from '
                             '(e.g. runs/foo/reward_predictor_checkpoints/'
                             'reward_predictor.ckpt-100)')

    parser.add_argument('--reward_predictor_ckpt_interval',
                        type=int, default=100,
                        help='No. training batches between reward '
                             'predictor checkpoints')

    # A2C options
    parser.add_argument('--load_policy_ckpt_dir',
                        help='Load a policy checkpoint from this directory.')
    parser.add_argument('--policy_ckpt_interval', type=int, default=100,
                        help="No. updates between policy checkpoints")

    args = parser.parse_args()

    params.init_params()
    params.params['test_mode'] = args.test_mode
    params.params['just_pretrain'] = args.just_pretrain
    params.params['debug'] = args.debug
    params.params['no_pretrain'] = args.no_pretrain
    params.params['save_prefs'] = args.save_prefs
    params.params['network'] = args.network
    params.params['just_prefs'] = args.just_prefs
    params.params['save_pretrain'] = args.save_pretrain
    params.params['print_lr'] = args.print_lr
    params.params['env'] = args.env
    params.params['orig_rewards'] = args.orig_rewards
    params.params['skip_prefs'] = args.skip_prefs
    params.params['batchnorm'] = args.batchnorm
    params.params['dropout'] = args.dropout
    params.params['n_preds'] = args.n_preds
    params.params['save_initial_prefs'] = args.save_initial_prefs
    params.params['random_query'] = args.random_queries
    params.params['no_gather_prefs'] = args.no_gather_prefs
    params.params['no_a2c'] = args.no_a2c
    params.params['render_episodes'] = args.render_episodes

    if args.test_mode:
        print("=== WARNING: running in test mode", file=sys.stderr)
        params.params['n_initial_prefs'] = 2
        params.params['n_initial_epochs'] = 1
        params.params['prefs_save_interval'] = 1
        params.params['reward_predictor_ckpt_interval'] = 1
        params.params['reward_predictor_val_interval'] = 1
        num_timesteps = 1000
    else:
        params.params['n_initial_prefs'] = args.n_initial_prefs
        params.params['n_initial_epochs'] = args.n_initial_epochs
        params.params['prefs_save_interval'] = 10
        params.params['reward_predictor_ckpt_interval'] = \
            args.reward_predictor_ckpt_interval
        params.params['reward_predictor_val_interval'] = 50
        num_timesteps = int(args.million_timesteps * 1e6)

    if args.env == 'MovingDotNoFrameskip-v0':
        params.params['policy'] = 'mlp'
    elif args.env == 'PongNoFrameskip-v4':
        params.params['policy'] = 'cnn'
    elif args.env == 'EnduroNoFrameskip-v4':
        params.params['policy'] = 'cnn'
    else:
        raise Exception("Policy unknown for env {}".format(args.env))

    if not osp.exists('.git'):
        git_rev = "unkrev"
    else:
        git_rev = subprocess.check_output(['git', 'rev-parse', '--short',
                                           'HEAD']).rstrip().decode()
    run_name = args.run_name + '_' + git_rev
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        log_dir = osp.join('runs', run_name)
        if osp.exists(log_dir):
            raise Exception("Log directory '%s' already exists" % log_dir)
        os.makedirs(log_dir)

    with open(osp.join(log_dir, 'args.txt'), 'w') as args_file:
        print(args, file=args_file)
        print(params.params, file=args_file)

    if args.lr_zero_million_timesteps is None:
        schedule = 'constant'
        nvalues = 1  # ignored
    else:
        schedule = 'linear'
        nvalues = int(args.lr_zero_million_timesteps * 1e6)
    lr_scheduler = Scheduler(v=args.lr,
                             nvalues=nvalues,
                             schedule=schedule)

    misc_logs_dir = osp.join(log_dir, 'misc')
    os.makedirs(misc_logs_dir)
    easy_tf_log.set_dir(misc_logs_dir)

    train(
        env_id=args.env,
        num_timesteps=num_timesteps,
        seed=args.seed,
        lr_scheduler=lr_scheduler,
        rp_lr=args.rp_lr,
        num_cpu=args.n_envs,
        rp_ckpt_path=args.load_reward_predictor_ckpt,
        load_prefs_dir=args.load_prefs_dir,
        headless=args.headless,
        log_dir=log_dir,
        ent_coef=args.ent_coef,
        db_max=args.db_max,
        segs_max=args.segs_max,
        log_interval=args.log_interval,
        policy_ckpt_dir=args.load_policy_ckpt_dir,
        policy_ckpt_interval=args.policy_ckpt_interval)


if __name__ == '__main__':
    main()
