#!/usr/bin/env python

import argparse
import os
from multiprocessing import Process

import numpy as np

import gym
import gym_gridworld  # noqa: F401 (imported but unused)
import params
from baselines.common.atari_wrappers import wrap_deepmind_nomax
from reward_predictor import RewardPredictorEnsemble

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


def f(cluster_dict, rp_ckpt_dir):
    RewardPredictorEnsemble(
        name='ps',
        cluster_dict=cluster_dict,
        load_network=False,
        rp_ckpt_dir=rp_ckpt_dir)


def test(rp):
    env = wrap_deepmind_nomax(gym.make("GridWorldNoFrameskip-v4"))
    nh, nw, nc = env.observation_space.shape
    nenvs = 1
    nstack = 4
    accuracies = []
    for i in range(10):
        print("Run {}:".format(i))
        obs = np.zeros((nenvs, nh, nw, nc*nstack), dtype=np.uint8)
        raw_obs = env.reset()
        obs = update_obs(obs, raw_obs, nc)
        done = False
        n_steps = 0
        n_right = 0
        rs = []
        obss = []
        while not done:
            action = env.unwrapped.action_space.sample()
            raw_obs, r, done, _ = env.step(action)
            rs.append(r)
            obs = update_obs(obs, raw_obs, nc)
            obs[:, 0, 0, -1] = 100 + action
            obss.append(np.copy(obs[0]))
            n_steps += 1

        rs_pred = rp.reward(np.array(obss))
        for r, r_pred in zip(rs, rs_pred):
            if np.sign(r) == np.sign(r_pred):
                n_right += 1
            print("{:+.1f} {:+.3f}".format(r, r_pred))
        print(n_right, n_steps)
        accuracies.append(n_right / n_steps)
    return np.mean(accuracies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')
    args = parser.parse_args()

    params.init_params()
    params.params['debug'] = False
    params.params['network'] = 'handcrafted'
    params.params['batchnorm'] = False

    cluster_dict = {
            'ps': ['localhost:2200'],
            'train_reward': ['localhost:2201']
            }
    Process(target=f, args=(cluster_dict, args.ckpt), daemon=True).start()
    rp = RewardPredictorEnsemble(
        name='train_reward',
        cluster_dict=cluster_dict,
        load_network=False,
        rp_ckpt_dir=args.ckpt)
    print(test(rp))


if __name__ == '__main__':
    main()
