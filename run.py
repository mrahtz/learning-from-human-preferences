#!/usr/bin/env python
import logging
import os
import pickle
from multiprocessing import Process, Queue
from threading import Thread
import time
import os.path as osp
import subprocess

import memory_profiler

import gym
import gym_gridworld
from baselines import bench, logger
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from pref_interface import PrefInterface
from reward_predictor import train_reward_predictor

def configure_logger(log_dir):
    baselines_dir = osp.join(log_dir, 'baselines')
    os.makedirs(baselines_dir)

    json_file = open(osp.join(baselines_dir, 'progress.json'), 'wt')
    json = logger.JSONOutputFormat(json_file)

    tb = logger.TensorBoardOutputFormat(baselines_dir)

    formats = [json, tb]
    logger.Logger.CURRENT = logger.Logger(dir=baselines_dir, output_formats=formats)


def train(env_id, num_frames, seed, lr, rp_lr, lrschedule, num_cpu,
        load_reward_network, load_prefs, headless, log_dir, ent_coef):
    configure_logger(log_dir)

    num_timesteps = int(num_frames / 4 * 1.1)
    # divide by 4 due to frameskip, then do a little extras so episodes end

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and
                osp.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    policy_fn = CnnPolicy

    # This has to be done here before any threads have been created by
    # TensorFlow, because it needs to spawn a GUI process (using fork()), and the
    # Objective C APIs (invoked when dealing with GUI stuff) aren't happy if
    # running in a processed forked from a multithreaded parent.
    pi = PrefInterface(headless)

    seg_pipe = Queue()
    pref_pipe = Queue()
    go_pipe = Queue(maxsize=1)
    a2c_proc = Process(target=lambda: learn(policy_fn, env, seed, seg_pipe,
        go_pipe, total_timesteps=num_timesteps, lr=lr, lrschedule=lrschedule,
        log_dir=log_dir, ent_coef=0.01), daemon=True)
    train_proc = Process(target=train_reward_predictor, args=(rp_lr, pref_pipe,
        go_pipe, load_reward_network, load_prefs, log_dir), daemon=True)

    a2c_proc.start()
    train_proc.start()

    """
    def profile(name, pid):
        with open(osp.join(log_dir, name + '.log'), 'w') as f:
            memory_profiler.memory_usage(pid, stream=f, timeout=99999)
    Process(target=profile, args=('a2c', a2c_proc.pid), daemon=True).start()
    Process(target=profile, args=('train', train_proc.pid), daemon=True).start()
    Process(target=profile, args=('interface', -1), daemon=True).start()
    """

    pi.run(seg_pipe, pref_pipe)

    while True:
        import time
        time.sleep(1.0)

    # TODO: this is never reached
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='GridWorldNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--lrschedule', help='Learning rate schedule',
            choices=['constant', 'linear'], default='linear')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--rp_lr', type=float, default=1e-4)
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=80)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--load_reward_network', action='store_true')
    parser.add_argument('--load_prefs', action='store_true')
    parser.add_argument('--headless', action='store_true', default=True)
    seconds_since_epoch = str(int(time.time()))
    parser.add_argument('--run_name', default=seconds_since_epoch)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    args = parser.parse_args()

    git_rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).rstrip().decode()
    run_name = args.run_name + '_' + git_rev
    log_dir = osp.join('runs', run_name)
    if osp.exists(log_dir):
        raise Exception("Log directory '%s' already exists" % log_dir)
    os.makedirs(log_dir)

    train(
        args.env,
        num_frames=1e6 * args.million_frames,
        seed=args.seed,
        lr=args.lr,
        rp_lr=args.rp_lr,
        lrschedule=args.lrschedule,
        num_cpu=args.n_envs,
        load_reward_network=args.load_reward_network,
        load_prefs=args.load_prefs,
        headless=args.headless,
        log_dir=log_dir,
        ent_coef=args.ent_coef)


if __name__ == '__main__':
    main()
