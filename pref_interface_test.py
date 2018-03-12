#!/usr/bin/env python3
import unittest
import numpy as np

from pref_db import Segment
from pref_interface import sample_seg_pair
from itertools import combinations


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


if __name__ == '__main__':
    unittest.main()
