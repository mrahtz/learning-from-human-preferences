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
import gym_gridworld  # noqa: F401 (imported but unused)
import params
sys.path.insert(0, 'baselines')
from baselines import bench, logger
from baselines.a2c.a2c import learn
from baselines.a2c.policies import MlpPolicy, CnnPolicy
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import wrap_deepmind_nomax
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble, train_reward_predictor
from enduro_wrapper import EnduroWrapper

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


def train(env_id, num_frames, seed, lr, rp_lr, lrschedule, num_cpu,
          rp_ckpt_dir, load_prefs_dir, headless, log_dir, ent_coef,
          db_max, segs_max, log_interval, policy_ckpt_dir):
    configure_logger(log_dir)

    num_timesteps = int(num_frames / 4 * 1.1)

    # divide by 4 due to frameskip, then do a little extras so episodes end

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

    parts_to_run = ['a2c', 'pref_interface', 'train_reward']

    if params.params['no_gather_prefs'] or params.params['orig_rewards']:
        parts_to_run.remove('pref_interface')
    if params.params['orig_rewards']:
        parts_to_run.remove('train_reward')
    if params.params['no_a2c']:
        parts_to_run.remove('a2c')

    port = 2200
    cluster_dict = {'ps': ['localhost:{}'.format(port)]}
    for part in parts_to_run:
        port += 1
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
            policy_fn,
            env,
            seed,
            seg_pipe,
            go_pipe,
            total_timesteps=num_timesteps,
            lr=lr,
            lrschedule=lrschedule,
            log_dir=log_dir,
            ent_coef=0.01,
            log_interval=log_interval,
            load_path=policy_ckpt_dir,
            reward_predictor=reward_predictor)

    def trp():
        reward_predictor = RewardPredictorEnsemble(
                name='train_reward',
                cluster_dict=cluster_dict,
                log_dir=log_dir)
        train_reward_predictor(
            reward_predictor=reward_predictor,
            lr=rp_lr,
            pref_pipe=pref_pipe,
            go_pipe=go_pipe,
            load_prefs_dir=load_prefs_dir,
            log_dir=log_dir,
            db_max=db_max,
            rp_ckpt_dir=rp_ckpt_dir)

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

        # We have to give PrefInterface the reward predictor /after/ init,
        # because init needs to spawn a GUI process (using fork()),
        # and the Objective C APIs (invoked when dealing with GUI stuff) aren't
        # happy if running in a processed forked from a multithreaded parent.
        reward_predictor = RewardPredictorEnsemble(
                name='pref_interface',
                cluster_dict=cluster_dict)
        pi.init_reward_predictor(reward_predictor)

        pi.run(seg_pipe, pref_pipe, segs_max)

    if 'train_reward' not in parts_to_run:
        go_pipe.put(True)

    if params.params['just_prefs'] or params.params['just_pretrain']:
        train_proc.join()
    elif a2c_proc:
        a2c_proc.join()
    else:
        raise Exception("Error: no parts to wait for?")

    env.close()  # Kill the SubprocVecEnv processes
    ps_proc.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env', help='environment ID', default='EnduroNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument(
        '--lrschedule',
        help='Learning rate schedule',
        choices=['constant', 'linear'],
        default='constant')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--rp_lr', type=float, default=2e-4)
    parser.add_argument(
        '--million_frames',
        help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip',
        type=int,
        default=40)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--rp_ckpt_dir')
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
    parser.add_argument('--policy_ckpt_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--orig_rewards', action='store_true')
    parser.add_argument('--skip_prefs', action='store_true')
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--n_preds', type=int, default=1)
    parser.add_argument('--save_initial_prefs', action='store_true')
    parser.add_argument('--random_queries', action='store_true')
    parser.add_argument('--no_gather_prefs', action='store_true')
    parser.add_argument('--no_a2c', action='store_true')
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

    if args.test_mode:
        print("=== WARNING: running in test mode", file=sys.stderr)
        params.params['n_initial_prefs'] = 2
        params.params['n_initial_epochs'] = 1
        params.params['save_freq'] = 1
        params.params['ckpt_freq'] = 1
    else:
        params.params['n_initial_prefs'] = 500
        params.params['n_initial_epochs'] = args.n_initial_epochs
        params.params['save_freq'] = 10
        params.params['ckpt_freq'] = 100

    if args.env == 'GridWorldNoFrameskip-v4':
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

    train(
        args.env,
        num_frames=1e6 * args.million_frames,
        seed=args.seed,
        lr=args.lr,
        rp_lr=args.rp_lr,
        lrschedule=args.lrschedule,
        num_cpu=args.n_envs,
        rp_ckpt_dir=args.rp_ckpt_dir,
        load_prefs_dir=args.load_prefs_dir,
        headless=args.headless,
        log_dir=log_dir,
        ent_coef=args.ent_coef,
        db_max=args.db_max,
        segs_max=args.segs_max,
        log_interval=args.log_interval,
        policy_ckpt_dir=args.policy_ckpt_dir)


if __name__ == '__main__':
    main()
