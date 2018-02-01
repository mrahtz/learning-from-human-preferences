#!/usr/bin/env python

import argparse

import cloudpickle
import gym
import numpy as np

import gym_gridworld  # noqa: F401 (imported but unused)
from baselines.common.atari_wrappers import wrap_deepmind_nomax


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


def main():
    parser = argparse.ArgumentParser(description="Run a trained model.")
    parser.add_argument("model", help="e.g. LOG_DIR/make_model.pkl")
    parser.add_argument("checkpoint", help="e.g. LOG_DIR/checkpoint100000")
    args = parser.parse_args()

    env = wrap_deepmind_nomax(gym.make("GridWorldNoFrameskip-v4"))

    with open(args.model, 'rb') as fh:
        make_model = cloudpickle.loads(fh.read())
    model = make_model()
    model.load(args.checkpoint)

    nenvs = 1
    nstack = int(model.step_model.X.shape[-1])
    nh, nw, nc = env.observation_space.shape
    obs = np.zeros((nenvs, nh, nw, nc*nstack), dtype=np.uint8)

    model_nenvs = int(model.step_model.X.shape[0])

    states = model.initial_state
    dones = [False]

    while True:
        raw_obs, dones[0] = env.reset(), False
        obs = update_obs(obs, raw_obs, nc)
        _obs = np.vstack([obs] * model_nenvs)
        episode_rew = 0
        actions, values, states = model.step(_obs, states, dones)
        action = actions[0]

        while not dones[0]:
            env.render()
            raw_obs, rew, dones[0], _ = env.step(action)
            obs = update_obs(obs, raw_obs, nc)
            _obs = np.vstack([obs] * model_nenvs)
            actions, values, states = model.step(_obs, states, dones)
            action = actions[0]
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
