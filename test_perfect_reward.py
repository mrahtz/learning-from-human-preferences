#!/usr/bin/env python

import numpy as np

import gym
import gym_gridworld  # noqa: F401 (imported but unused)
from baselines.common.atari_wrappers import wrap_deepmind_nomax


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


def main():
    env = wrap_deepmind_nomax(gym.make("GridWorldNoFrameskip-v4"))

    for i in range(5):
        os = []
        o = env.reset()
        os.append(o)
        done = False
        rewards = []
        while not done:
            pos_pre = env.unwrapped.pos[:]
            steps_pre = env.unwrapped.steps
            rs = []
            for action in range(env.unwrapped.action_space.n):
                _, r, _, _ = env.step(action)
                rs.append(r)
                env.unwrapped.pos[:] = pos_pre
                env.unwrapped.steps = steps_pre
            action = np.argmax(rs)
            o, r, done, _ = env.step(action)
            rewards.append(r)
            os.append(o)
            env.render()
        print(sum(rewards))


if __name__ == '__main__':
    main()
