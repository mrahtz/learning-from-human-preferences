#!/usr/bin/env python3
import time
import unittest
from itertools import combinations
from multiprocessing import Queue

import numpy as np
import termcolor

from pref_db import Segment
from pref_interface import PrefInterface


def send_segments(n_segments, seg_pipe):
    frame_stack = np.zeros((84, 84, 4))
    for i in range(n_segments):
        segment = Segment()
        for _ in range(25):
            segment.append(frame=frame_stack, reward=0)
        segment.finalise(seg_id=i)
        seg_pipe.put(segment)


class TestPrefInterface(unittest.TestCase):
    def setUp(self):
        self.p = PrefInterface(synthetic_prefs=True, max_segs=1000,
                               log_dir='/tmp')
        termcolor.cprint(self._testMethodName, 'red')

    def testSampleSegPair(self):
        seg_pipe = Queue()
        n_segments = 5
        send_segments(n_segments, seg_pipe)
        self.p.recv_segments(seg_pipe)

        # Check that we get exactly the right number of unique pairs back
        n_possible_pairs = len(list(combinations(range(n_segments), 2)))
        tested_pairs = set()
        for _ in range(n_possible_pairs):
            s1, s2 = self.p.sample_seg_pair()
            tested_pairs.add((s1.hash, s2.hash))
            tested_pairs.add((s2.hash, s1.hash))
        self.assertEqual(len(tested_pairs), 2 * n_possible_pairs)

        # Check that if we try to get just one more, we get an exception
        # indicating that there are no more unique pairs available
        with self.assertRaises(IndexError):
            self.p.sample_seg_pair()

    def testRecvSegments(self):
        pi = PrefInterface(synthetic_prefs=True, max_segs=5, log_dir='/tmp')
        pipe = Queue()
        for i in range(5):
            pipe.put(i)
            pi.recv_segments(pipe)
        np.testing.assert_array_equal(pi.segments, [0, 1, 2, 3, 4])
        for i in range(5, 8):
            pipe.put(i)
            pi.recv_segments(pipe)
        np.testing.assert_array_equal(pi.segments, [5, 6, 7, 3, 4])
        for i in range(8, 11):
            pipe.put(i)
            pi.recv_segments(pipe)
        np.testing.assert_array_equal(pi.segments, [10, 6, 7, 8, 9])


if __name__ == '__main__':
    unittest.main()
