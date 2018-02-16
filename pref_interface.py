#!/usr/bin/env python

import queue
import time
from itertools import combinations
from multiprocessing import Process, Queue

import numpy as np
import pyglet
from numpy.testing import assert_equal

import params
from reward_predictor import RewardPredictorEnsemble
from scipy.ndimage import zoom


class Im(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (
            self.height,
            self.width), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            self.width, self.height, 'L', arr.tobytes(), pitch=-self.width)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


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
        pair_preds = self.reward_model.preferences(s1s, s2s)
        pair_preds = np.array(pair_preds)
        n_preds = self.reward_model.n_preds
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

    def sample_segments(self, segments):
        if len(segments) <= 10:
            idxs = range(len(segments))
        else:
            n_segs = len(segments)
            idxs = np.random.choice(range(n_segs),
                                    size=10, replace=False)
        return idxs

    def ask_user(self, s1, s2):
        border = np.zeros((84, 10), dtype=np.uint8)

        seg_len = len(s1)
        vid = []
        for t in range(seg_len):
            # Show only the most recent frame of the 4-frame stack
            frame = np.hstack((s1[t][:, :, -1], border, s2[t][:, :, -1]))
            frame = zoom(frame, 2)
            vid.append(frame)
        n_pause_frames = 7
        vid.extend([vid[-1]] * n_pause_frames)
        self.vid_q.put(vid)

        while True:
            print("Choice: ")
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
        tested_idxs = set()
        segments = []
        self.reward_model = RewardPredictorEnsemble('pref_interface')

        while True:
            self.recv_segments(segments, seg_pipe, segs_max)
            if len(segments) >= 2:
                break
            print("Not enough segments yet; sleeping...")
            time.sleep(1.0)

        while True:
            pair_idxs = []
            # Get a list of all possible pairs of segments, then filter out the
            # pairs we've already tested
            while len(pair_idxs) == 0:
                self.recv_segments(segments, seg_pipe, segs_max)
                idxs = self.sample_segments(segments)
                pair_idxs = set(combinations(idxs, 2))
                pair_idxs = pair_idxs - tested_idxs
                pair_idxs = list(pair_idxs)
            (n1, n2), s1, s2 = self.least_certain_seg_pair(segments, pair_idxs)
            if params.params['debug']:
                print("Querying preference for segments", n1, "and", n2)

            if not self.synthetic_prefs:
                pref = self.ask_user(s1.frames, s2.frames)
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

            pref_pipe.put((s1, s2, pref))
            tested_idxs.add((n1, n2))


def vid_proc(q):
    v = Im()
    segment = q.get(block=True, timeout=None)
    t = 0
    while True:
        v.imshow(segment[t])
        try:
            segment = q.get(block=False)
            if segment == "Pause":
                segment = q.get(block=True)
            t = 0
        except queue.Empty:
            t = (t + 1) % len(segment)
            time.sleep(1/15)
