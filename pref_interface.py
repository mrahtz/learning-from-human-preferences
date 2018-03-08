#!/usr/bin/env python
import datetime
import queue
import time
from itertools import combinations
from multiprocessing import Process, Queue
from utils import save_pref_db

import numpy as np
from numpy.testing import assert_equal

from scipy.ndimage import zoom

from utils import vid_proc, PrefDB

import os.path as osp
import logging
import sys
import os


class PrefInterface:

    def __init__(self, headless, synthetic_prefs=False):
        self.vid_q = Queue()
        if not headless:
            Process(target=vid_proc, args=(self.vid_q,), daemon=True).start()
        self.synthetic_prefs = synthetic_prefs

    def recv_segments(self, segments, seg_pipe, segs_max):
        n_segs = 0
        while True:
            try:
                segment = seg_pipe.get(timeout=0.1)
                n_segs += 1
            except queue.Empty:
                break
            segments.append(segment)
            # (The maximum number of segments kept being 5,000 isn't mentioned
            # in the paper anywhere - it's just something I decided on. This
            # should be maximum ~ 700 MB.)
            if len(segments) > segs_max:
                del segments[0]

    def sample_pair_idxs(self, segments, exclude_pairs):
        # Page 14: "[We] draw a factor 10 more clip pair candidates than we
        # ultimately present to the human."
        possible_pairs = combinations(range(len(segments)), 2)
        possible_pairs = list(set(possible_pairs) - set(exclude_pairs))

        if len(possible_pairs) <= 10:
            return possible_pairs
        else:
            idxs = np.random.choice(range(len(possible_pairs)),
                                    size=10, replace=False)
            pairs = []
            for i in idxs:
                pairs.append(possible_pairs[i])
            return pairs

    def ask_user(self, n1, n2, s1, s2):
        border = np.zeros((84, 10), dtype=np.uint8)

        seg_len = len(s1)
        vid = []
        for t in range(seg_len):
            # Show only the most recent frame of the 4-frame stack
            frame = np.hstack((s1[t][:, :, -1], border, s2[t][:, :, -1]))
            frame = zoom(frame, 4)
            vid.append(frame)
        n_pause_frames = 7
        vid.extend([vid[-1]] * n_pause_frames)
        self.vid_q.put(vid)

        while True:
            print("Segments {} and {}: ".format(n1, n2))
            choice = input()
            if choice == "L" or choice == "R" or choice == "E" or choice == "":
                break
            else:
                print("Invalid choice '{}'".format(choice))

        if choice == "L":
            pref = (1., 0.)
        elif choice == "R":
            pref = (0., 1.)
        elif choice == "E":
            pref = (0.5, 0.5)
        elif choice == "":
            pref = None

        self.vid_q.put("Pause")

        return pref

    def run(self, seg_pipe, pref_pipe, segs_max):
        tested_pairs = []
        segments = []

        while True:
            self.recv_segments(segments, seg_pipe, segs_max)
            if len(segments) >= 2:
                break
            print("Waiting for segments")
            time.sleep(2.0)

        while True:
            pair_idxs = []
            # If we've tested all the possible pairs of segments so far,
            # we might have to wait
            while len(pair_idxs) == 0:
                self.recv_segments(segments, seg_pipe, segs_max)
                pair_idxs = self.sample_pair_idxs(segments,
                                                  exclude_pairs=tested_pairs)
            logging.debug("Sampled segment pairs:", pair_idxs)

            i = np.random.choice(len(pair_idxs))
            (n1, n2) = pair_idxs[i]
            s1 = segments[n1]
            s2 = segments[n2]

            logging.debug("Querying preference for segments", n1, "and", n2)

            if not self.synthetic_prefs:
                pref = self.ask_user(n1, n2, s1.frames, s2.frames)
            else:
                if sum(s1.rewards) > sum(s2.rewards):
                    pref = (1.0, 0.0)
                elif sum(s1.rewards) < sum(s2.rewards):
                    pref = (0.0, 1.0)
                else:
                    pref = (0.5, 0.5)

            # We don't need the rewards from this point on
            s1 = s1.frames
            s2 = s2.frames

            if pref is not None:
                pref_pipe.put((s1, s2, pref))
            tested_pairs.append((n1, n2))
            tested_pairs.append((n2, n1))


def recv_prefs(pref_pipe, pref_db_train, pref_db_val, db_max):
    n_recvd = 0
    val_fraction = 0.2
    while True:
        try:
            s1, s2, mu = pref_pipe.get(timeout=0.1)
            n_recvd += 1
        except queue.Empty:
            break

        if np.random.rand() < val_fraction:
            pref_db_val.append(s1, s2, mu)
        else:
            pref_db_train.append(s1, s2, mu)

        if len(pref_db_val) > db_max * val_fraction:
            pref_db_val.del_first()
        assert len(pref_db_val) <= db_max * val_fraction

        if len(pref_db_train) > db_max * (1 - val_fraction):
            pref_db_train.del_first()
        assert len(pref_db_train) <= db_max * (1 - val_fraction)


def get_initial_prefs(pref_pipe, n_initial_prefs, db_max):
    pref_db_val = PrefDB()
    pref_db_train = PrefDB()
    # Page 15: "We collect 500 comparisons from a randomly initialized policy
    # network at the beginning of training"
    while len(pref_db_train) < n_initial_prefs or len(pref_db_val) == 0:
        print("Waiting for preferences; %d so far" % len(pref_db_train))
        recv_prefs(pref_pipe, pref_db_train, pref_db_val, db_max)
        time.sleep(5.0)

    return pref_db_train, pref_db_val

def save_prefs(pref_db_train, pref_db_val, save_dir, name):
    fname = osp.join(save_dir, "train_{}.pkl".format(name))
    save_pref_db(pref_db_train, fname)
    fname = osp.join(save_dir, "val_{}.pkl".format(name))
    save_pref_db(pref_db_val, fname)
