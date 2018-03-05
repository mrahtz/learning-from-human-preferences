#!/usr/bin/env python

import queue
import time
from itertools import combinations
from multiprocessing import Process, Queue

import numpy as np
from numpy.testing import assert_equal

import params
from scipy.ndimage import zoom
from utils import vid_proc


class PrefInterface:

    def __init__(self, headless, synthetic_prefs=False):
        self.vid_q = Queue()
        if not headless:
            Process(target=vid_proc, args=(self.vid_q,), daemon=True).start()
        self.synthetic_prefs = synthetic_prefs

    def least_certain_seg_pair(self, segments, pair_idxs):
        """
        - Calculate predicted preferences for every possible pair of segments
        - Calculate the pair with the highest uncertainty
        - Return the index of that pair of segments
        - Send that pair of segments, along with segment IDs, to be checked by
          the user
        """
        s1s = []
        s2s = []
        for i1, i2 in pair_idxs:
            s1s.append(segments[i1].frames)
            s2s.append(segments[i2].frames)
        pair_preds = self.reward_predictor.preferences(s1s, s2s)
        pair_preds = np.array(pair_preds)
        n_preds = self.reward_predictor.n_preds
        assert_equal(pair_preds.shape, (n_preds, len(pair_idxs), 2))

        # Each predictor gives two outputs:
        # - p1: the probability of segment 1 being preferred
        # - p2: the probability of segment 2 being preferred
        #       (= 1 - p1)
        # We want to calculate variance of predictions across all
        # predictors in the ensemble.
        # If L is a list, var(L) = var(1 - L).
        # So we can calculate the variance based on either p1 or p2
        # and get the same result.
        preds = pair_preds[:, :, 0]
        assert_equal(preds.shape, (n_preds, len(pair_idxs)))

        # Calculate variances across ensemble members
        pred_vars = np.var(preds, axis=0)
        assert_equal(pred_vars.shape, (len(pair_idxs), ))

        highest_var_i = np.argmax(pred_vars)
        check_idxs = pair_idxs[highest_var_i]
        check_s1 = segments[check_idxs[0]]
        check_s2 = segments[check_idxs[1]]

        return check_idxs, check_s1, check_s2

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

    def init_reward_predictor(self, reward_predictor):
        self.reward_predictor = reward_predictor

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
            if params.params['debug']:
                print("Sampled segment pairs:")
                print(pair_idxs)

            if not params.params['random_query']:
                # TODO: this /might/ be currently broken
                # e.g. try running
                # ./run.py --n_initial_prefs 1 --headless
                (n1, n2), s1, s2 = \
                    self.least_certain_seg_pair(segments, pair_idxs)
            else:
                i = np.random.choice(len(pair_idxs))
                (n1, n2) = pair_idxs[i]
                s1 = segments[n1]
                s2 = segments[n2]

            if params.params['debug']:
                print("Querying preference for segments", n1, "and", n2)

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
