#!/usr/bin/env python

"""
Run a trained checkpoint to see what the agent is actually doing in the
environment.
"""

import argparse
import os.path as osp
import time
from collections import deque

import cloudpickle
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import FormatStrFormatter

from enduro_wrapper import EnduroWrapper
from openai_baselines.common.atari_wrappers import wrap_deepmind_nomax


def main():
    args = parse_args()

    env = get_env(args.env)
    model = get_model(args.policy_ckpt_dir)
    if args.reward_predictor_ckpt_dir:
        reward_predictor = get_reward_predictor(args.reward_predictor_ckpt_dir)
    else:
        reward_predictor = None

    run_agent(env, model, reward_predictor, args.frame_interval_ms)


def run_agent(env, model, reward_predictor, frame_interval_ms):
    nenvs = 1
    nstack = int(model.step_model.X.shape[-1])
    nh, nw, nc = env.observation_space.shape
    obs = np.zeros((nenvs, nh, nw, nc * nstack), dtype=np.uint8)
    model_nenvs = int(model.step_model.X.shape[0])
    states = model.initial_state
    if reward_predictor:
        value_graph = ValueGraph()
    while True:
        raw_obs = env.reset()
        update_obs(obs, raw_obs, nc)
        episode_reward = 0
        done = False
        while not done:
            model_obs = np.vstack([obs] * model_nenvs)
            [action], _, states = model.step(model_obs, states, [done])
            raw_obs, reward, done, _ = env.step(action)
            update_obs(obs, raw_obs, nc)
            episode_reward += reward
            env.render()
            if reward_predictor is not None:
                predicted_reward = reward_predictor.reward(obs)
                # reward_predictor.reward returns reward for each frame in the
                # supplied batch. We only supplied one frame, so get the reward
                # for that frame.
                value_graph.append(predicted_reward[0])
            time.sleep(frame_interval_ms * 1e-3)
        print("Episode reward:", episode_reward)


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs


def get_reward_predictor(ckpt_dir):
    with open(osp.join(ckpt_dir, 'make_reward_predictor.pkl'), 'rb') as fh:
        make_reward_predictor = cloudpickle.loads(fh.read())
    cluster_dict = {'a2c': ['localhost:2200']}
    print("Initialising reward predictor...")
    reward_predictor = make_reward_predictor(name='a2c', cluster_dict=cluster_dict)
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    reward_predictor.init_network(ckpt_file)
    return reward_predictor


def get_model(ckpt_dir):
    model_file = osp.join(ckpt_dir, 'make_model.pkl')
    with open(model_file, 'rb') as fh:
        make_model = cloudpickle.loads(fh.read())
    print("Initialising policy...")
    model = make_model()
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    print("Loading checkpoint...")
    model.load(ckpt_file)
    return model


def get_env(env_id):
    env = gym.make(env_id)
    if env_id == 'EnduroNoFrameskip-v4':
        env = EnduroWrapper(env)
    elif env_id == 'MovingDot-v0' or env_id == 'MovingDotNoFrameskip-v0':
        pass
    env = wrap_deepmind_nomax(env)
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env")
    parser.add_argument("policy_ckpt_dir")
    parser.add_argument("--reward_predictor_ckpt_dir")
    parser.add_argument("--frame_interval_ms", type=float, default=0.)
    args = parser.parse_args()
    return args


class ValueGraph:
    def __init__(self):
        n_values = 100
        self.data = deque(maxlen=n_values)

        self.fig, self.ax = plt.subplots()
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.fig.set_size_inches(4, 2)
        self.ax.set_xlim([0, n_values - 1])
        self.ax.grid(axis='y')  # Draw a line at 0 reward
        self.y_min = float('inf')
        self.y_max = -float('inf')
        self.line, = self.ax.plot([], [])

        self.fig.show()
        self.fig.canvas.draw()

    def append(self, value):
        self.data.append(value)

        self.y_min = min(self.y_min, min(self.data))
        self.y_max = max(self.y_max, max(self.data))
        self.ax.set_ylim([self.y_min, self.y_max])
        self.ax.set_yticks([self.y_min, 0, self.y_max])
        plt.tight_layout()

        ydata = list(self.data)
        xdata = list(range(len(self.data)))
        self.line.set_data(xdata, ydata)

        self.ax.draw_artist(self.line)
        self.fig.canvas.draw()


if __name__ == '__main__':
    main()
