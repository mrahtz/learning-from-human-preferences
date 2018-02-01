#!/usr/bin/env python3

import unittest
from itertools import combinations

import numpy as np

import gym
import gym_gridworld  # noqa: F401 (imported but unused)
from baselines.common.atari_wrappers import wrap_deepmind_nomax
from dot_utils import predict_action_rewards  # noqa: F401 (imported butunused)
from dot_utils import predict_action_preference


class TestDotUtils(unittest.TestCase):


    def test_predict_action_preferences(self):
        """
        Simulate a few episodes of the agent wandering around randomly and
        check whether the reward predictor's preferences about episodes is
        correct.
        """
        env = wrap_deepmind_nomax(gym.make("GridWorldNoFrameskip-v4"))
        episodes = []
        episode_rewards = []

        def update_obs(obs_stack, obs):
            obs_stack = np.roll(obs_stack, shift=-1, axis=3)
            obs_stack[:, :, :, -1] = obs[:, :, :, 0]
            return obs_stack

        for _ in range(5):
            nenv = 1
            nh = 84
            nw = 84
            nc = 1
            nstack = 4
            obs_stack = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)

            segment = []
            obs = env.reset()
            obs = np.expand_dims(obs, 0)  # nenv dimension
            obs_stack = update_obs(obs_stack, obs)
            done = False
            rewards = []
            while not done:
                # The reward predictor might be wrong when the dot is close to
                # the middle because of errors introduced by the frame scaling.
                # Therefore we shouldn't use a uniform distribution, because it
                # could easily get net zero reward (from being equally likely
                # to take steps in good directions as bad direction). Instead
                # we use a linear distribution.
                a_probs = np.arange(start=1, stop=(env.action_space.n + 1))
                a_probs = a_probs / sum(a_probs)
                action = np.random.choice(np.arange(5), p=a_probs)
                obs_stack[0, 0, 0, -1] = 100 + action
                segment.append(obs_stack[0])

                obs, r, done, _ = env.step(action)
                obs = np.expand_dims(obs, 0)
                obs_stack = update_obs(obs_stack, obs)
                rewards.append(r)

            episode_rewards.append(rewards)
            episodes.append(np.copy(segment))

        for i1, i2 in list(combinations(range(len(episodes)), 2)):
            e1, e2 = episodes[i1], episodes[i2]

            er1 = episode_rewards[i1]
            er2 = episode_rewards[i2]
            """
            par1 = predict_action_rewards(e1)
            par2 = predict_action_rewards(e2)
            print("Expected:", sum(er1), sum(er2))
            print("Actual", sum(par1), sum(par2))
            print(er1)
            print(par1)
            """

            actual_pref = predict_action_preference(e1, e2)
            if sum(er1) > sum(er2):
                expected_pref = (1.0, 0.0)
            elif sum(er1) < sum(er2):
                expected_pref = (0.0, 1.0)
            else:
                expected_pref = (0.5, 0.5)
            self.assertEqual(actual_pref, expected_pref)


if __name__ == '__main__':
    unittest.main()
