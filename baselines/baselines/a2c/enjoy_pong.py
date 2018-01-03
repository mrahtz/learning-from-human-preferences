import argparse

import cloudpickle
import gym
import numpy as np

from baselines.a2c.a2c import Model
from baselines.a2c.policies import CnnPolicy
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


def main():
    parser = argparse.ArgumentParser(description="Run a trained model.")
    parser.add_argument("model", help="e.g. LOG_DIR/make_model.pkl")
    parser.add_argument("checkpoint", help="e.g. LOG_DIR/checkpoint100000")
    args = parser.parse_args()

    env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))

    with open(args.model, 'rb') as fh:
        make_model = cloudpickle.loads(fh.read())
    model = make_model()
    model.load(args.checkpoint)

    nenvs = 1
    nstack = int(model.step_model.X.shape[-1])
    nh, nw, nc = env.observation_space.shape
    obs = np.zeros((nenvs, nh, nw, nc*nstack), dtype=np.uint8)

    states = model.initial_state
    dones = [False]

    while True:
        raw_obs, done = env.reset(), False
        obs = update_obs(obs, raw_obs, nc)
        episode_rew = 0
        [action], [value], states = model.step(obs, states, dones)

        while not done:
            env.render()
            raw_obs, rew, dones[0], _ = env.step(action)
            obs = update_obs(obs, raw_obs, nc)
            [action], [value], states = model.step(obs, states, dones)
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
