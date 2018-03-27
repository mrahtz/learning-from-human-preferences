#!/usr/bin/env python3
import unittest

import tensorflow as tf
import termcolor
import numpy as np
from numpy import exp, log
from numpy.testing import (assert_allclose, assert_approx_equal,
                           assert_array_equal, assert_raises)

from reward_predictor import RewardPredictorNetwork
from reward_predictor_core_network import net_cnn


class TestRewardPredictor(unittest.TestCase):

    def setUp(self):
        self.create_reward_predictor(dropout=0.5, batchnorm=True)
        termcolor.cprint(self._testMethodName, 'red')

    def create_reward_predictor(self, dropout, batchnorm):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.rpn = RewardPredictorNetwork(batchnorm=batchnorm, dropout=dropout,
                                          lr=1e-3,
                                          core_network=net_cnn)
        self.sess.run(tf.global_variables_initializer())

    def test_weight_sharing(self):
        """
        Check that both legs of the network give the same reward output
        for the same segment input.
        """

        # We deliberately /don't/ use the same dropout for each leg of the
        # network. (If we do use the same dropout, without batchnorm,
        # Pong doesn't train successfully. If we use different dropout, Pong
        # does train successfully. I haven't tried training Pong with
        # batchnorm.) So we disable dropout for this test.
        self.create_reward_predictor(dropout=0.0, batchnorm=True)

        s = 255 * np.random.rand(100, 84, 84, 4)
        feed_dict_nontraining = {
            self.rpn.s1: [s],
            self.rpn.s2: [s],
            self.rpn.training: True
        }
        feed_dict_training = {
            self.rpn.s1: [s],
            self.rpn.s2: [s],
            self.rpn.training: False
        }
        for feed_dict in [feed_dict_nontraining, feed_dict_training]:
            for _ in range(3):  # to check different dropouts
                [rs1], [rs2] = self.sess.run([self.rpn.rs1, self.rpn.rs2], feed_dict)
                # Check rs1 != 0.0
                assert_raises(AssertionError, assert_array_equal, rs1, 0.0)
                assert_allclose(rs1, rs2)

    def test_batchnorm_sharing(self):
        """
        Check that batchnorm statistics are the same between the two legs of
        the network.
        """
        n_frames = 20
        s1 = 255 * np.random.normal(loc=1.0, size=(n_frames, 84, 84, 4))
        s2 = 255 * np.random.normal(loc=-1.0, size=(n_frames, 84, 84, 4))
        feed_dict = {
            self.rpn.s1: [s1],
            self.rpn.s2: [s2],
            self.rpn.pref: [[0.0, 1.0]],
            self.rpn.training: True}
        self.sess.run(self.rpn.train, feed_dict)

        feed_dict = {self.rpn.s1: [s1], self.rpn.s2: [s1], self.rpn.training: False}
        [rs1], [rs2] = self.sess.run([self.rpn.rs1, self.rpn.rs2], feed_dict)
        # Check rs1 != 0.0
        assert_raises(AssertionError, assert_array_equal, rs1, 0.0)
        assert_allclose(rs1, rs2)

    def test_loss(self):
        """
        Check that the loss is calculated correctly.
        """
        # hack to ensure numerical stability
        rs1 = rs2 = 100
        n_frames = 20
        while rs1 > 50 or rs2 > 50:
            s1 = 255 * np.random.normal(loc=1.0, size=(n_frames, 84, 84, 4))
            s2 = 255 * np.random.normal(loc=-1.0, size=(n_frames, 84, 84, 4))
            feed_dict = {
                self.rpn.s1: [s1],
                self.rpn.s2: [s2],
                self.rpn.training: True
            }
            [rs1], [rs2] = self.sess.run([self.rpn.rs1, self.rpn.rs2],
                                         feed_dict)

        prefs = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        for pref in prefs:
            feed_dict[self.rpn.pref] = [pref]
            [rs1], [rs2], loss = self.sess.run(
                [self.rpn.rs1, self.rpn.rs2, self.rpn.loss], feed_dict)

            p_s1_s2 = exp(rs1) / (exp(rs1) + exp(rs2))
            p_s2_s1 = exp(rs2) / (exp(rs1) + exp(rs2))

            expected = -(pref[0] * log(p_s1_s2) + pref[1] * log(p_s2_s1))
            assert_approx_equal(loss, expected, significant=3)

    def test_batches(self):
        """
        Present a batch of two trajectories and check that we get the same
        results as if we'd presented the trajectories individually.
        """
        n_segs = 2
        n_frames = 20
        prefs = [[0., 1.], [1., 0.]]
        s1s = []
        s2s = []
        for _ in range(n_segs):
            s1 = 255 * np.random.normal(loc=1.0, size=(n_frames, 84, 84, 4))
            s2 = 255 * np.random.normal(loc=-1.0, size=(n_frames, 84, 84, 4))
            s1s.append(s1)
            s2s.append(s2)

        # Step 1: present all trajectories as one big batch
        feed_dict = {
            self.rpn.s1: s1s,
            self.rpn.s2: s2s,
            self.rpn.pref: prefs,
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
                self.rpn.pref: [prefs[i]],
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

        Note: because of variations in training, this test does not always pass.
        """
        n_frames = 20
        s1 = 255 * np.random.normal(loc=1.0, size=(n_frames, 84, 84, 4))
        s2 = 255 * np.random.normal(loc=-1.0, size=(n_frames, 84, 84, 4))

        feed_dict = {
            self.rpn.s1: [s1],
            self.rpn.s2: [s2]
        }

        prefs = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        for pref in prefs:
            print("Preference", pref)
            feed_dict[self.rpn.pref] = [pref]
            # Important to reset batch normalization statistics
            self.sess.run(tf.global_variables_initializer())
            for _ in range(150):
                feed_dict[self.rpn.training] = True
                self.sess.run(self.rpn.train, feed_dict)
                # Uncomment these for more thorough manual testing.
                # (For the first case, rs1 should become higher
                #  than rs2, and the distance between them should increase;
                #  for the second case, rs2 should become higher;
                #  for the third case, they should become approximately the
                #  same.)
            """
                feed_dict[self.rpn.training] = False
                ops = [self.rpn.rs1, self.rpn.rs2, self.rpn.loss]
                [rs1], [rs2], loss = self.sess.run(ops, feed_dict)
                print(" ".join(3 * ["{:>8.3f}"]).format(rs1, rs2, loss))
            print()
            """

            feed_dict[self.rpn.training] = False
            [rs1], [rs2] = self.sess.run([self.rpn.rs1, self.rpn.rs2], feed_dict)

            if pref[0] > pref[1]:
                self.assertGreater(rs1 - rs2, 10)
            elif pref[1] > pref[0]:
                self.assertGreater(rs2 - rs1, 10)
            elif pref[0] == pref[1]:
                self.assertLess(abs(rs2 - rs1), 2)

    def test_training_batches(self):
        """
        Check that after training with a batch of 4 segments, each with their own preferences,
        the predicted preference for each of the segments is as expected.
        """
        n_frames = 20
        s1s = 255 * np.random.normal(loc=1.0, size=(4, n_frames, 84, 84, 4))
        s2s = 255 * np.random.normal(loc=-1.0, size=(4, n_frames, 84, 84, 4))
        prefs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        feed_dict = {
            self.rpn.s1: s1s,
            self.rpn.s2: s2s,
            self.rpn.pref: prefs,
            self.rpn.training: True
        }

        for i in range(100):
            if i % 10 == 0:
                print("Training {}/100".format(i))
            self.sess.run(self.rpn.train, feed_dict)

        feed_dict[self.rpn.training] = False
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
        s1s = 255 * np.random.normal(loc=1.0, size=(batch_n, n_frames, 84, 84, 4))
        s2s = 255 * np.random.normal(loc=-1.0, size=(batch_n, n_frames, 84, 84, 4))
        possible_prefs = [[1.0, 0.0], [0.0, 1.0]]
        possible_prefs = np.array(possible_prefs)
        prefs = possible_prefs[np.random.choice([0, 1], size=batch_n)]

        feed_dict = {
            self.rpn.s1: s1s,
            self.rpn.s2: s2s,
            self.rpn.pref: prefs,
            self.rpn.training: True
        }

        # Steer away from chance performance
        for _ in range(5):
            self.sess.run(self.rpn.train, feed_dict)

        feed_dict[self.rpn.training] = False
        preds = self.sess.run(self.rpn.pred, feed_dict)
        n_correct = 0
        for pref, pred in zip(prefs, preds):
            if pref[0] == 1.0 and pred[0] > pred[1] or \
               pref[1] == 1.0 and pred[1] > pred[0]:
                n_correct += 1
        accuracy_expected = n_correct / batch_n

        accuracy_actual = self.sess.run(self.rpn.accuracy, feed_dict)

        assert_approx_equal(accuracy_actual, accuracy_expected)


if __name__ == '__main__':
    unittest.main()
