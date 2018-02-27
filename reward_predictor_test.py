#!/usr/bin/env python3
import unittest

import numpy as np
from numpy import exp, log
from numpy.testing import (assert_allclose, assert_approx_equal,
                           assert_array_equal, assert_raises)

import gym
import gym_moving_dot  # noqa: F401 (imported but unused)
import params
import tensorflow as tf
from baselines.common.atari_wrappers import wrap_deepmind_nomax
from reward_predictor import RewardPredictor, batch_iter, get_position


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


class TestRewardPredictor(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        #self.rpn = RewardPredictor(batchnorm=False, dropout=0.5, lr=1e-3)
        self.sess.run(tf.global_variables_initializer())
        params.init_params()
        params.params['network'] = 'handcrafted'

    def test_batch_iter(self):
        l1 = list(range(16))
        l2 = list(range(15))

        for l in [l1, l2]:
            expected = l
            actual = []
            for x in batch_iter(l, batch_size=4, shuffle=False):
                actual.extend(x)
            np.testing.assert_array_equal(actual, expected)

            expected = l
            actual = []
            for x in batch_iter(l, batch_size=4, shuffle=True):
                actual.extend(x)
            actual = np.sort(actual)
            np.testing.assert_array_equal(actual, expected)

    def test_weight_sharing(self):
        """
        Input a trajectory, and test that both networks give the same
        reward output.
        """
        s = np.random.rand(100, 84, 84, 4)

        feed_dict = {
            self.rpn.s1: [s],
            self.rpn.s2: [s],
            self.rpn.dropout: 0.0,
            self.rpn.training: False
        }
        [rs1], [rs2] = self.sess.run([self.rpn.rs1, self.rpn.rs2], feed_dict)
        assert_allclose(rs1, rs2)

    def test_loss(self):
        """
        Input two trajectories, and check that the loss is calculated
        correctly.
        """
        # hack to ensure numerical stability
        rs1 = rs2 = 100
        n_frames = 20
        while rs1 > 50 or rs2 > 50:
            s1 = np.random.normal(loc=1.0, size=(n_frames, 84, 84, 4))
            s2 = np.random.normal(loc=-1.0, size=(n_frames, 84, 84, 4))
            feed_dict = {
                self.rpn.s1: [s1],
                self.rpn.s2: [s2],
                self.rpn.dropout: 0.0,
                self.rpn.training: True
            }
            [rs1], [rs2] = self.sess.run([self.rpn.rs1, self.rpn.rs2],
                                         feed_dict)

        mus = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        for mu in mus:
            feed_dict[self.rpn.mu] = [mu]
            [rs1], [rs2], loss = self.sess.run(
                [self.rpn.rs1, self.rpn.rs2, self.rpn.loss], feed_dict)

            p_s1_s2 = exp(rs1) / (exp(rs1) + exp(rs2))
            p_s2_s1 = exp(rs2) / (exp(rs1) + exp(rs2))

            expected = -(mu[0] * log(p_s1_s2) + mu[1] * log(p_s2_s1))
            assert_approx_equal(loss, expected, significant=3)

    def test_batches(self):
        """
        Present a batch of two trajectories and check that we get the same
        results as if we'd presented the trajectories individually.
        """
        n_segs = 2
        n_frames = 20
        mus = [[0., 1.], [1., 0.]]
        s1s = []
        s2s = []
        for _ in range(n_segs):
            s1 = np.random.normal(loc=1.0, size=(n_frames, 84, 84, 4))
            s2 = np.random.normal(loc=-1.0, size=(n_frames, 84, 84, 4))
            s1s.append(s1)
            s2s.append(s2)

        # Step 1: present all trajectories as one big batch
        feed_dict = {
            self.rpn.s1: s1s,
            self.rpn.s2: s2s,
            self.rpn.mu: mus,
            self.rpn.dropout: 0.0,
            self.rpn.training: False
        }
        rs1_batch, rs2_batch, pred_batch, loss_batch = self.sess.run(
            [self.rpn.rs1, self.rpn.rs2, self.rpn.pred, self.rpn.loss],
            feed_dict)

        # Step 2: present trajectories individually
        rs1_nobatch = []
        rs2_nobatch = []
        pred_nobatch = []
        loss_nobatch = 0
        for i in range(n_segs):
            feed_dict = {
                self.rpn.s1: [s1s[i]],
                self.rpn.s2: [s2s[i]],
                self.rpn.mu: [mus[i]],
                self.rpn.dropout: 0.0,
                self.rpn.training: False
            }
            [rs1], [rs2], [pred], loss = self.sess.run(
                [self.rpn.rs1, self.rpn.rs2, self.rpn.pred, self.rpn.loss],
                feed_dict)
            rs1_nobatch.append(rs1)
            rs2_nobatch.append(rs2)
            pred_nobatch.append(pred)
            loss_nobatch += loss

        # Compare
        assert_allclose(rs1_batch, rs1_nobatch, atol=1e-5)
        assert_allclose(rs2_batch, rs2_nobatch, atol=1e-5)
        assert_allclose(pred_batch, pred_nobatch, atol=1e-5)
        assert_approx_equal(loss_batch, loss_nobatch, significant=4)

    def test_training(self):
        """
        Present two trajectories with different preferences and see whether
        training really does work (whether the reward predicted by the network
        matches the preferences after a few loops of running the training
        operation).
        """
        n_frames = 20
        s1 = np.random.normal(loc=1.0, size=(n_frames, 84, 84, 4))
        s2 = np.random.normal(loc=-1.0, size=(n_frames, 84, 84, 4))

        feed_dict = {
            self.rpn.s1: [s1],
            self.rpn.s2: [s2],
            self.rpn.dropout: 0.0,
            self.rpn.training: True
        }

        mus = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        for mu in mus:
            feed_dict[self.rpn.mu] = [mu]
            for i in range(10):
                [rs1], [rs2], [pred], loss = self.sess.run(
                    [self.rpn.rs1, self.rpn.rs2, self.rpn.pred, self.rpn.loss],
                    feed_dict)
                # Uncomment this for more thorough manual testing.
                # (For the first case, rs1 should become higher
                #  than rs2, and the distance between them should increase;
                #  for the second case, rs2 should become higher;
                #  for the third case, they should become approximately the
                #  same.)
                #print(rs1, rs2, loss)
                self.sess.run(self.rpn.train, feed_dict)
            #print()

            if mu[0] > mu[1]:
                self.assertGreater(rs1, rs2)
            elif mu[1] > mu[0]:
                self.assertGreater(rs2, rs1)
            elif mu[0] == mu[1]:
                # TODO tweak this once variance normalization is implemented
                # TODO this doesn't work consistently, but eh
                self.assertLess(abs(rs2 - rs1), 40)

    def test_training_batches(self):
        n_frames = 20
        s1s = np.random.normal(loc=1.0, size=(4, n_frames, 84, 84, 4))
        s2s = np.random.normal(loc=-1.0, size=(4, n_frames, 84, 84, 4))
        mus = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        feed_dict = {
            self.rpn.s1: s1s,
            self.rpn.s2: s2s,
            self.rpn.mu: mus,
            self.rpn.dropout: 0.0,
            self.rpn.training: True
        }

        for i in range(10):
            self.sess.run(self.rpn.train, feed_dict)
        preds = self.sess.run(self.rpn.pred, feed_dict)
        assert_allclose(preds[0], [1., 0.], atol=1e-1)
        assert_allclose(preds[1], [1., 0.], atol=1e-1)
        assert_allclose(preds[2], [0., 1.], atol=1e-1)
        assert_allclose(preds[3], [0., 1.], atol=1e-1)

    def test_accuracy(self):
        """
        Test accuracy op.
        """
        n_frames = 20
        batch_n = 16
        s1s = np.random.normal(loc=1.0, size=(batch_n, n_frames, 84, 84, 4))
        s2s = np.random.normal(loc=-1.0, size=(batch_n, n_frames, 84, 84, 4))
        possible_mus = [[1.0, 0.0], [0.0, 1.0]]
        possible_mus = np.array(possible_mus)
        mus = possible_mus[np.random.choice([0, 1], size=batch_n)]

        feed_dict = {
            self.rpn.s1: s1s,
            self.rpn.s2: s2s,
            self.rpn.mu: mus,
            self.rpn.dropout: 0.0,
            self.rpn.training: False
        }

        # Steer away from chance performance
        for i in range(5):
            self.sess.run(self.rpn.train, feed_dict)

        preds = self.sess.run(self.rpn.pred, feed_dict)
        n_correct = 0
        for mu, pred in zip(mus, preds):
            if mu[0] == 1.0 and pred[0] > pred[1] or \
               mu[1] == 1.0 and pred[1] > pred[0]:
                n_correct += 1
        accuracy_expected = n_correct / batch_n

        accuracy_actual = self.sess.run(self.rpn.accuracy, feed_dict)

        assert_approx_equal(accuracy_actual, accuracy_expected)

    def test_handcrafted(self):
        """
        Test whether the handcrafted reward function can extract the position
        of the dot correctly
        """
        env = wrap_deepmind_nomax(gym.make('MovingDotNoFrameskip-v0'))

        nh, nw, nc = env.observation_space.shape
        nenvs = 1
        nstack = 4
        obs = np.zeros((nenvs, nh, nw, nc * nstack), dtype=np.uint8)

        s = tf.placeholder(tf.float32, (None, 84, 84, 4))
        xo, yo = get_position(s)

        for episode_n in range(5):
            raw_obs = env.reset()
            epos1 = np.copy(env.unwrapped.pos)
            obs = update_obs(obs, raw_obs, nc)
            x, y = self.sess.run([xo, yo], feed_dict={s: obs})
            pos1 = np.array([x, y]).flatten()

            done = False
            while not done:
                action = env.unwrapped.action_space.sample()
                raw_obs, r, done, _ = env.step(action)
                epos2 = env.unwrapped.pos

                if done:
                    # If done, then we probably won't have taken the full 4
                    # steps we usually take (because of wrap_deepmind), so our
                    # position may not have changed enough to be visible in the
                    # observations downscaled to 84 x 84
                    # (also by wrap_deepmind)
                    break

                obs = update_obs(obs, raw_obs, nc)
                obs[:, 0, 0, -1] = 100 + action

                x, y = self.sess.run([xo, yo], feed_dict={s: obs})
                pos2 = np.array([x, y]).flatten()

                for i in [0, 1]:
                    if epos2[i] < epos1[i]:
                        self.assertLess(pos2[i], pos1[i])
                    elif epos2[i] == epos1[i]:
                        self.assertEqual(pos2[i], pos1[i])
                    else:
                        self.assertGreater(pos2[i], pos1[i])

                epos1 = np.copy(epos2)
                pos1 = np.copy(pos2)


if __name__ == '__main__':
    unittest.main()
