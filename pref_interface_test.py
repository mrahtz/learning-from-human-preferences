#!/usr/bin/env python3
import unittest
from itertools import combinations
from multiprocessing import Queue

import numpy as np

from pref_db import Segment
from pref_interface import sample_seg_pair, PrefInterface


class TestPrefInterface(unittest.TestCase):
    def testSampleSegPairs(self):
        # Make a small array of segments
        segments = []
        frame_stack = np.zeros((84, 84, 4))
        for i in range(5):
            segment = Segment()
            for _ in range(25):
                segment.append(frame=frame_stack, reward=0)
            segment.finalise(seg_id=i)
            segments.append(segment)

        # Check that we get exactly the right number of unique pairs back
        n_possible_pairs = len(list(combinations(range(len(segments)), 2)))
        tested_pairs = set()
        for _ in range(n_possible_pairs):
            s1, s2 = sample_seg_pair(segments, tested_pairs)
            tested_pairs.add((s1.hash, s2.hash))
            tested_pairs.add((s2.hash, s1.hash))
        self.assertEqual(len(tested_pairs), 2 * n_possible_pairs)

        # Check that if we try to get just one more, we get an exception
        # indicating that there are no more unique pairs available
        with self.assertRaises(IndexError):
            sample_seg_pair(segments, tested_pairs)

    def testRecvSegments(self):
        pi = PrefInterface(headless=True, max_segs=5)
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
