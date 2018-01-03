#!/usr/bin/env python

import os.path
import pickle
import queue
import time
from itertools import combinations
from multiprocessing import Process, Queue
from threading import Thread

import numpy as np
import pyglet
from numpy.testing import assert_equal

from dot_utils import predict_preference
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
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'L', arr.tobytes(), pitch=-self.width)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()


class PrefInterface:

    def __init__(self, headless):
        self.vid_q = Queue()
        if not headless:
            Process(target=vid_proc, args=(self.vid_q,), daemon=True).start()

    def get_seg_pair(self):
        """
        - Calculate predicted preferences for every possible pair of segments
        - Calculate the pair with the highest uncertainty
        - Return the index of that pair of segments
        - Send that pair of segments, along with segment IDs, to be checked by
          the user
        """
        if len(self.segments) < 2:
            raise Exception("Not enough segments yet")

        if len(self.segments) > 10:
            idxs = np.random.choice(
                range(len(self.segments)), size=10, replace=False)
        else:
            idxs = range(len(self.segments))

        print("Calculating preferences...")
        s1s = []
        s2s = []
        pair_idxs = list(combinations(idxs, 2))
        for i1, i2 in pair_idxs:
            s1s.append(self.segments[i1])
            s2s.append(self.segments[i2])
        pair_preds = self.reward_model.preferences(s1s, s2s)
        pair_preds = np.array(pair_preds)
        print("done!")

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
        seg_0_preds = pair_preds[:, :, 0]
        assert_equal(seg_0_preds.shape, (n_preds, len(pair_idxs)))
        # Calculate variances across ensemble members
        pair_vars = np.var(seg_0_preds, axis=0)
        assert_equal(pair_vars.shape, (len(pair_idxs), ))
        highest_var_i = np.argmax(pair_vars)
        # Select this pair to show to the user
        # TODO check more carefully
        check_idxs = pair_idxs[highest_var_i]
        check_s1 = self.segments[check_idxs[0]]
        check_s2 = self.segments[check_idxs[1]]
        print("Pair with highest variance is", check_idxs)
        print("Predictions are:")
        print(seg_0_preds[:, highest_var_i])

        # TODO: loop if already checked
        return check_idxs, check_s1, check_s2

    def recv_segments(self, seg_pipe):
        while True:
            segment = seg_pipe.get()
            self.segments.append(segment)

            # (The maximum number of segments kept being 5,000 isn't mentioned
            # in the paper anywhere - it's just something I decided on. This
            # should be maximum ~ 700 MB.)
            if len(self.segments) > 1000:
                del self.segments[0]
            assert len(self.segments) <= 1000

    def run(self, seg_pipe, pref_pipe):
        self.segments = []
        self.reward_model = RewardPredictorEnsemble('pref_interface')
        Thread(target=self.recv_segments, args=(seg_pipe, )).start()

        while True:
            try:
                (n1, n2), s1, s2 = self.get_seg_pair()
                break
            except Exception as e:
                print(e)
                time.sleep(1.0)
                continue

        while True:
            mu = predict_preference(s1, s2)
            print("Segment pair %d and %d:" % (n1, n2), mu)
            pref_pipe.put((s1, s2, mu))
            (n1, n2), s1, s2 = self.get_seg_pair()


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
