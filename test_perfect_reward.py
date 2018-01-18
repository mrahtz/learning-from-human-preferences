#!/usr/bin/env python

import numpy as np

import gym
import gym_gridworld
from baselines.common.atari_wrappers import wrap_deepmind
from dot_utils import predict_reward


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


def rew(env):
    middle = np.array([160/2, 210/2])
    d = np.linalg.norm(env.unwrapped.pos - middle)
    return -d


def main():
    env = wrap_deepmind(gym.make("GridWorldNoFrameskip-v4"))

    for i in range(5):
        env.reset()
        done = False
        os = []
        while not done:
            pos_pre = env.unwrapped.pos[:]
            steps_pre = env.unwrapped.steps
            rs = []
            for action in range(env.unwrapped.action_space.n):
                o, _, s, _ = env.step(action)
                r = rew(env)
                rs.append(r)
                env.unwrapped.pos[:] = pos_pre
                env.unwrapped.steps = steps_pre
            action = np.argmax(rs)
            #action = env.unwrapped.action_space.sample()
            o, r, done, _ = env.step(action)
            os.append(o)
            #env.render()
        print(predict_reward(os))


if __name__ == '__main__':
    main()
