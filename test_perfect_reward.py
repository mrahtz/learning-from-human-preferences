#!/usr/bin/env python

import numpy as np

import gym
import gym_gridworld
from baselines.common.atari_wrappers import wrap_deepmind
from dot_utils import predict_action_rewards


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


def main():
    env = wrap_deepmind(gym.make("GridWorldNoFrameskip-v4"))

    for i in range(5):
        os = []
        o = env.reset()
        os.append(o)
        done = False
        actual_rs = []
        actions = []
        while not done:
            pos_pre = env.unwrapped.pos[:]
            steps_pre = env.unwrapped.steps
            rs = []
            for action in range(env.unwrapped.action_space.n):
                _, r, _, _ = env.step(action)
                rs.append(r)
                env.unwrapped.pos[:] = pos_pre
                env.unwrapped.steps = steps_pre
            # action = np.argmax(rs)
            action = env.action_space.sample()
            actions.append(env.unwrapped.get_action_meanings()[action])
            o, r, done, _ = env.step(action)
            actual_rs.append(r)
            os.append(o)
            # env.render()
        # Also test new synthetic rewards
        predicted_rs = predict_action_rewards(os)
        n_errors = sum([abs(a - b) for a, b in zip(predicted_rs, actual_rs)])
        print("{}/{} wrong".format(n_errors, len(predicted_rs)))
        print(sum(predict_action_rewards(os)))


if __name__ == '__main__':
    main()
