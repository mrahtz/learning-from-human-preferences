#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
import logging
import os
import os.path as osp
import subprocess
import time
from multiprocessing import Queue

import gym
import gym_gridworld
import params
from baselines import bench, logger
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, NnPolicy, MlpPolicy
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def configure_logger(log_dir):
    baselines_dir = osp.join(log_dir, 'baselines')
    os.makedirs(baselines_dir)

    json_file = open(osp.join(baselines_dir, 'progress.json'), 'wt')
    json = logger.JSONOutputFormat(json_file)

    tb = logger.TensorBoardOutputFormat(baselines_dir)

    formats = [json, tb]
    logger.Logger.CURRENT = logger.Logger(
        dir=baselines_dir, output_formats=formats)


def train(env_id, num_frames, seed, policy, lrschedule, num_cpu, log_dir):
    configure_logger(log_dir)
    num_timesteps = int(num_frames / 4 * 1.1)

    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(
                env,
                logger.get_dir() and os.path.join(
                    logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)

        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'nn':
        policy_fn = NnPolicy
    elif policy == 'mlp':
        policy_fn = MlpPolicy
    seg_pipe = Queue()
    go_pipe = Queue()
    go_pipe.put(True)
    learn(
        policy_fn,
        env,
        seed,
        seg_pipe,
        go_pipe,
        total_timesteps=num_timesteps,
        lrschedule=lrschedule,
        orig_rewards=True,
        gen_segs=False,
        log_dir=log_dir,
        log_interval=10)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env', help='environment ID', default='GridWorldNoFrameskip-v4')
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument(
        '--policy',
        help='Policy architecture',
        choices=['cnn', 'lstm', 'lnlstm', 'nn', 'mlp'],
        default='mlp')
    parser.add_argument(
        '--lrschedule',
        help='Learning rate schedule',
        choices=['constant', 'linear'],
        default='constant')
    parser.add_argument(
        '--million_frames',
        help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip',
        type=int,
        default=40)
    seconds_since_epoch = str(int(time.time()))
    parser.add_argument('--run_name', default=seconds_since_epoch)
    parser.add_argument('--log_dir')
    args = parser.parse_args()

    params.init_params()
    params.params['debug'] = True
    params.params['print_lr'] = False
    params.params['env'] = args.env

    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        git_rev = subprocess.check_output(['git', 'rev-parse', '--short',
                                           'HEAD']).rstrip().decode()
        run_name = args.run_name + '_' + git_rev
        log_dir = osp.join('runs', run_name)
        if osp.exists(log_dir):
            raise Exception("Log directory '%s' already exists" % log_dir)
        os.makedirs(log_dir)

    train(
        args.env,
        num_frames=1e6 * args.million_frames,
        seed=args.seed,
        policy=args.policy,
        lrschedule=args.lrschedule,
        num_cpu=args.n_envs,
        log_dir=log_dir)


if __name__ == '__main__':
    main()
