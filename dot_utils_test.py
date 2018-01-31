#!/usr/bin/env python3

import unittest
from itertools import combinations

import numpy as np

import gym
import gym_gridworld  # noqa: F401 (imported but unused)
from baselines.common.atari_wrappers import wrap_deepmind_nomax
from dot_utils import predict_action_preference, predict_action_rewards
from dot_utils import get_coords


class TestDotUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_predict_action_preferences(self):
        env = wrap_deepmind_nomax(gym.make("GridWorldNoFrameskip-v4"))

        segments = []
        r_sums = []
        for run_n in range(5):
            segment = []
            obs = env.reset()
            segment.append(obs)
            done = False
            actual_rs = []
            while not done:
                a_probs = np.arange(start=1, stop=6)
                a_probs = a_probs / sum(a_probs)
                action = np.random.choice(np.arange(5), p=a_probs)
                o, r, done, _ = env.step(action)
                segment.append(o)

                action_name = env.unwrapped.get_action_meanings()[action]
                print("I took action", action_name, "and now I'm at",
                      env.unwrapped.pos)
                print(np.sum(o))
                print("Predict thinks I'm at", get_coords(o))

                predicted_r = predict_action_rewards(segment[-2:])
                print(r, predicted_r)
                """
                if r != int(predicted_r[0]):
                    print("ERR")
                    from IPython.core.debugger import Pdb
                    Pdb().set_trace()
                """

                actual_rs.append(r)
            predicted_rs = predict_action_rewards(segment)
            print(actual_rs[:10])
            print(predict_action_rewards(segment)[:10])
            n_errs = sum([abs(a - b) for a, b in zip(predicted_rs, actual_rs)])
            print(n_errs)
            r_sums.append(sum(actual_rs))
            segments.append(np.copy(segment))

        for i1, i2 in list(combinations(range(len(segments)), 2)):
            print("Segments %d and %d" % (i1, i2))
            print("Expected:", r_sums[i1], r_sums[i2])
            s1, s2 = segments[i1], segments[i2]
            print("Actual:", sum(predict_action_rewards(s1)),
                  sum(predict_action_rewards(s2)))
            actual_mu = predict_action_preference(s1, s2)
            if r_sums[i1] > r_sums[i2]:
                expected_mu = (1.0, 0.0)
            elif r_sums[i1] < r_sums[i2]:
                expected_mu = (0.0, 1.0)
            else:
                expected_mu = (0.5, 0.5)
            self.assertEqual(actual_mu, expected_mu)


if __name__ == '__main__':
    unittest.main()
