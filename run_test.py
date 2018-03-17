#!/usr/bin/env python3

"""
Simple tests to make sure each of the main commands basically run fine without
any errors.
"""

import subprocess
import tempfile
import unittest
from os.path import exists, join

import termcolor


def create_initial_prefs(out_dir, interactive):
    cmd = ("python3 run.py gather_initial_prefs "
           "PongNoFrameskip-v4 "
           "--n_initial_prefs 1 "
           "--log_dir {}".format(out_dir))
    if not interactive:
        cmd += " --synthetic_prefs"
    subprocess.call(cmd.split(' '))


class TestRun(unittest.TestCase):

    def setUp(self):
        termcolor.cprint(self._testMethodName, 'red')

    def test_gather_prefs(self):
        for interactive in [True, False]:
            if interactive:
                termcolor.cprint('Human preferences', 'green')
            else:
                termcolor.cprint('Synthetic preferences', 'green')
            # Automatically deletes the directory afterwards :)
            with tempfile.TemporaryDirectory() as temp_dir:
                create_initial_prefs(temp_dir, interactive)
                self.assertTrue(exists(join(temp_dir, 'train_initial.pkl.gz')))
                self.assertTrue(exists(join(temp_dir, 'val_initial.pkl.gz')))

    def test_pretrain_reward_predictor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_initial_prefs(temp_dir, interactive=False)
            cmd = ("python3 run.py pretrain_reward_predictor "
                   "PongNoFrameskip-v4 "
                   "--n_initial_epochs 1 "
                   "--load_prefs_dir {0} "
                   "--log_dir {0}".format(temp_dir))
            subprocess.call(cmd.split(' '))
            self.assertTrue(exists(join(temp_dir,
                                        'reward_predictor_checkpoints',
                                        'reward_predictor.ckpt.index')))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = ("python3 run.py train_policy_with_preferences "
                   "PongNoFrameskip-v4 "
                   "--synthetic_prefs "
                   "--million_timesteps 0.0001 "
                   "--n_initial_prefs 1 "
                   "--n_initial_epochs 1 "
                   "--log_dir {0}".format(temp_dir))
            subprocess.call(cmd.split(' '))
            self.assertTrue(exists(join(temp_dir,
                                        'policy_checkpoints',
                                        'policy.ckpt.index')))
            self.assertTrue(exists(join(temp_dir,
                                        'reward_predictor_checkpoints',
                                        'make_reward_predictor.pkl')))


if __name__ == '__main__':
    unittest.main()
