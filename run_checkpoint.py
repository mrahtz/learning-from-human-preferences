#!/usr/bin/env python

import argparse
import os.path as osp
import sys
import time
from collections import deque

import cloudpickle
import gym_moving_dot  # noqa: F401 (imported but unused)
import numpy as np
import tensorflow as tf

import gym
import matplotlib
import params
from enduro_wrapper import EnduroWrapper
from matplotlib.ticker import FormatStrFormatter
from reward_predictor import RewardPredictorEnsemble

from openai_baselines.common.atari_wrappers import wrap_deepmind_nomax

if True:  # To silence flake8 warnings about imports not at top
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


class ValueGraph:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.fig.set_size_inches(4, 2)
        self.ax.set_xlim([0, 100])
        self.ax.grid(axis='y')  # Draw a line at 0 reward
        self.y_min = float('inf')
        self.y_max = -float('inf')
        self.line, = self.ax.plot([], [])

        self.fig.show()
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def plot(self, value_log):
        self.y_min = min(self.y_min, min(value_log))
        self.y_max = max(self.y_max, max(value_log))
        self.ax.set_ylim([self.y_min, self.y_max])
        self.ax.set_yticks([self.y_min, 0, self.y_max])
        plt.tight_layout()

        ydata = list(value_log)
        xdata = list(range(len(value_log)))
        self.line.set_data(xdata, ydata)

        self.ax.draw_artist(self.line)
        self.fig.canvas.draw()


def main():
    parser = argparse.ArgumentParser(description="Run a trained model.")
    parser.add_argument("ckpt_dir")
    parser.add_argument("--env", default='MovingDotNoFrameskip-v0')
    parser.add_argument("--reward_predictor_ckpt")
    parser.add_argument("--frame_interval_ms", type=float, default=0.)
    args = parser.parse_args()

    # Set up environment

    env = gym.make(args.env)
    if args.env == 'EnduroNoFrameskip-v4':
        env = EnduroWrapper(env)
    env = wrap_deepmind_nomax(env)
    env.unwrapped.maxsteps = 500

    model_file = osp.join(args.ckpt_dir, 'make_model.pkl')
    with open(model_file, 'rb') as fh:
        make_model = cloudpickle.loads(fh.read())
    print("Initialising policy...")
    model = make_model()

    ckpt_file = tf.train.latest_checkpoint(args.ckpt_dir)
    print("Loading checkpoint...")
    model.load(ckpt_file)

    nenvs = 1
    nstack = int(model.step_model.X.shape[-1])
    nh, nw, nc = env.observation_space.shape
    obs = np.zeros((nenvs, nh, nw, nc * nstack), dtype=np.uint8)

    model_nenvs = int(model.step_model.X.shape[0])

    states = model.initial_state
    dones = [False]

    # Set up reward predictor

    if args.reward_predictor_ckpt is None:
        reward_predictor = None
    else:
        params.init_params()
        # TODO: ideally these should be saved along with the checkpoint
        params.params['batchnorm'] = False
        params.params['dropout'] = 0.0
        params.params['n_preds'] = 1
        params.params['network'] = 'conv'
        params.params['debug'] = False
        cluster_dict = {'train_reward': ['localhost:2200']}
        reward_predictor = RewardPredictorEnsemble(
            name='train_reward',
            cluster_dict=cluster_dict,
            ckpt_path=args.reward_predictor_ckpt,
            log_dir='/tmp')
        value_log = deque(maxlen=100)
        value_graph = ValueGraph()

    while True:
        raw_obs, dones[0] = env.reset(), False
        obs = update_obs(obs, raw_obs, nc)
        _obs = np.vstack([obs] * model_nenvs)
        episode_rew = 0
        actions, values, states = model.step(_obs, states, dones)
        action = actions[0]

        while not dones[0]:
            env.render()
            if reward_predictor is not None:
                predicted_reward = reward_predictor.reward(obs)
                # reward_predictor.reward returns reward for each frame in the
                # supplied batch. We only supplied one frame, so get the reward
                # for that frame.
                predicted_reward = predicted_reward[0]
                value_log.append(predicted_reward)
                value_graph.plot(value_log)
            raw_obs, rew, dones[0], _ = env.step(action)
            obs = update_obs(obs, raw_obs, nc)
            _obs = np.vstack([obs] * model_nenvs)
            actions, values, states = model.step(_obs, states, dones)
            action = actions[0]
            episode_rew += rew
            time.sleep(args.frame_interval_ms * 1e-3)
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
