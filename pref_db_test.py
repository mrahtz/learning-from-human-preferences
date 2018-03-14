#!/usr/bin/env python3

import unittest
import numpy as np
from pref_db import PrefDB


class TestPrefDB(unittest.TestCase):

    def test_similar_segs(self):
        """
        Test that the preference database really distinguishes
        between similar segments
        (i.e. check that its hash function is working as it's supposed to).
        """
        p = PrefDB()
        s1 = np.ones((25, 84, 84, 4))
        s2 = np.ones((25, 84, 84, 4))
        s2[12][24][24][2] = 0
        p.append(s1, s2, [1.0, 0.0])
        self.assertEqual(len(p.segments), 2)

    def test_append_delete(self):
        """
        Do a number of appends/deletes and check that the number of
        preferences and segments is as expected at all times.
        """
        p = PrefDB()

        s1 = np.random.randint(low=-10, high=10, size=(25, 84, 84, 4))
        s2 = np.random.randint(low=-10, high=10, size=(25, 84, 84, 4))
        p.append(s1, s2, [1.0, 0.0])
        self.assertEqual(len(p.segments), 2)
        self.assertEqual(len(p.prefs), 1)

        p.append(s1, s2, [0.0, 1.0])
        self.assertEqual(len(p.segments), 2)
        self.assertEqual(len(p.prefs), 2)

        s1 = np.random.randint(low=-10, high=10, size=(25, 84, 84, 4))
        p.append(s1, s2, [1.0, 0.0])
        self.assertEqual(len(p.segments), 3)
        self.assertEqual(len(p.prefs), 3)

        s2 = np.random.randint(low=-10, high=10, size=(25, 84, 84, 4))
        p.append(s1, s2, [1.0, 0.0])
        self.assertEqual(len(p.segments), 4)
        self.assertEqual(len(p.prefs), 4)

        s1 = np.random.randint(low=-10, high=10, size=(25, 84, 84, 4))
        s2 = np.random.randint(low=-10, high=10, size=(25, 84, 84, 4))
        p.append(s1, s2, [1.0, 0.0])
        self.assertEqual(len(p.segments), 6)
        self.assertEqual(len(p.prefs), 5)

        prefs_pre = list(p.prefs)
        p.del_first()
        self.assertEqual(len(p.prefs), 4)
        self.assertEqual(p.prefs, prefs_pre[1:])
        # These segments were also used by the second preference,
        # so the number of segments shouldn't have decreased
        self.assertEqual(len(p.segments), 6)

        p.del_first()
        self.assertEqual(len(p.prefs), 3)
        # One of the segments just deleted was only used by the first two
        # preferences, so the length should have shrunk by one
        self.assertEqual(len(p.segments), 5)

        p.del_first()
        self.assertEqual(len(p.prefs), 2)
        # Another one should bite the dust...
        self.assertEqual(len(p.segments), 4)

        p.del_first()
        self.assertEqual(len(p.prefs), 1)
        self.assertEqual(len(p.segments), 2)

        p.del_first()
        self.assertEqual(len(p.prefs), 0)
        self.assertEqual(len(p.segments), 0)


if __name__ == '__main__':
    unittest.main()
