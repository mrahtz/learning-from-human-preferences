#!/usr/bin/env python

import socket
import unittest
from math import ceil

import numpy as np

from pref_db import PrefDB
from utils import RunningStat, get_port_range, batch_iter


class TestUtils(unittest.TestCase):

    # https://github.com/joschu/modular_rl/blob/master/modular_rl/running_stat.py
    def test_running_stat(self):
        for shp in ((), (3, ), (3, 4)):
            li = []
            rs = RunningStat(shp)
            for i in range(5):
                val = np.random.randn(*shp)
                rs.push(val)
                li.append(val)
                m = np.mean(li, axis=0)
                assert np.allclose(rs.mean, m)
                if i == 0:
                    continue
                # ddof=1 => calculate unbiased sample variance
                v = np.var(li, ddof=1, axis=0)
                assert np.allclose(rs.var, v)

    def test_similar_segs(self):
        """
        Test that the preference database really distinguishes
        between very similar segments (i.e. check that its hash function
        is working as it's supposed to).
        """
        p = PrefDB()
        s1 = np.ones((25, 84, 84, 4))
        s2 = np.ones((25, 84, 84, 4))
        s2[12][24][24][2] = 0
        p.append(s1, s2, [1.0, 0.0])
        self.assertEqual(len(p.segments), 2)

    def test_pref_db(self):
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

    def test_get_port_range(self):
        # Test 1: if we ask for 3 ports starting from port 60000
        # (which nothing should be listening on), we should get back
        # 60000, 60001 and 60002
        ports = get_port_range(60000, 3)
        self.assertEqual(ports, [60000, 60001, 60002])

        # Test 2: if we set something listening on port 60000
        # then ask for the same ports as in test 1,
        # the function should skip over 60000 and give us the next
        # three ports
        s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s1.bind(("127.0.0.1", 60000))
        ports = get_port_range(60000, 3)
        self.assertEqual(ports, [60001, 60002, 60003])

        # Test 3: if we set something listening on port 60002,
        # the function should realise it can't allocate a continuous
        # range starting from 60000 and should give us a range starting
        # from 60003
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.bind(("127.0.0.1", 60002))
        ports = get_port_range(60000, 3)
        self.assertEqual(ports, [60003, 60004, 60005])

        s2.close()
        s1.close()

    def test_batch_iter_1(self):
        """
        Check that batch_iter gives us exactly the right data back.
        """
        l1 = list(range(16))
        l2 = list(range(15))
        l3 = list(range(13))
        for l in [l1, l2, l3]:
            for shuffle in [True, False]:
                expected_data = l
                actual_data = set()
                expected_n_batches = ceil(len(l) / 4)
                actual_n_batches = 0
                for batch_n, x in enumerate(batch_iter(l, batch_size=4, shuffle=shuffle)):
                    if batch_n == expected_n_batches - 1 and len(l) % 4 != 0:
                        self.assertEqual(len(x), len(l) % 4)
                    else:
                        self.assertEqual(len(x), 4)
                    self.assertEqual(len(actual_data.intersection(set(x))), 0)
                    actual_data = actual_data.union(set(x))
                    actual_n_batches += 1
                self.assertEqual(actual_n_batches, expected_n_batches)
                np.testing.assert_array_equal(list(actual_data), expected_data)

    def test_batch_iter_2(self):
        """
        Check that shuffle=True returns the same data but in a different order.
        """
        expected_data = list(range(16))
        actual_data = []
        for x in batch_iter(expected_data, batch_size=4, shuffle=True):
            actual_data.extend(x)
        self.assertEqual(len(actual_data), len(expected_data))
        self.assertEqual(set(actual_data), set(expected_data))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(actual_data, expected_data)

    def test_batch_iter_3(self):
        """
        Check that successive calls shuffle in a different order.
        """
        data = list(range(16))
        out1 = []
        for x in batch_iter(data, batch_size=4, shuffle=True):
            out1.extend(x)
        out2 = []
        for x in batch_iter(data, batch_size=4, shuffle=True):
            out2.extend(x)
        self.assertEqual(set(out1), set(out2))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(out1, out2)


if __name__ == '__main__':
    unittest.main()

