#!/usr/bin/env python

"""
A simple CLI-based interface for querying the user about segment preferences.
"""

import logging
import queue
import time
from itertools import combinations
from multiprocessing import Queue
from random import shuffle

import easy_tf_log
import numpy as np

from utils import VideoRenderer


class PrefInterface:

    def __init__(self, synthetic_prefs, max_segs, log_dir):
        self.vid_q = Queue()
        if not synthetic_prefs:
            self.renderer = VideoRenderer(vid_queue=self.vid_q, zoom_factor=4)
        else:
            self.renderer = None
        self.synthetic_prefs = synthetic_prefs
        self.seg_idx = 0
        self.segments = []
        self.tested_pairs = set()  # For O(1) lookup
        self.max_segs = max_segs
        easy_tf_log.set_dir(log_dir)

    def stop_renderer(self):
        if self.renderer:
            self.renderer.stop()

    def run(self, seg_pipe, pref_pipe):
        while len(self.segments) < 2:
            print("Preference interface waiting for segments")
            time.sleep(5.0)
            self.recv_segments(seg_pipe)

        while True:
            seg_pair = None
            while seg_pair is None:
                try:
                    seg_pair = self.sample_seg_pair()
                except IndexError:
                    # If we've tested all possible pairs of segments so far,
                    # we'll have to wait for more segments
                    time.sleep(1.0)
                    self.recv_segments(seg_pipe)
                else:
                    break
            s1, s2 = seg_pair

            logging.debug("Querying preference for segments %s and %s",
                          s1.hash, s2.hash)

            if not self.synthetic_prefs:
                pref = self.ask_user(s1, s2)
            else:
                if sum(s1.rewards) > sum(s2.rewards):
                    pref = (1.0, 0.0)
                elif sum(s1.rewards) < sum(s2.rewards):
                    pref = (0.0, 1.0)
                else:
                    pref = (0.5, 0.5)

            if pref is not None:
                # We don't need the rewards from this point on, so just send
                # the frames
                pref_pipe.put((s1.frames, s2.frames, pref))
            # If pref is None, the user answered "incomparable" for the segment
            # pair. The pair has been marked as tested; we just drop it.

    def recv_segments(self, seg_pipe):
        """
        Receive segments from `seg_pipe` into circular buffer `segments`.
        """
        while True:
            try:
                segment = seg_pipe.get(timeout=0.1)
            except queue.Empty:
                break

            if len(self.segments) < self.max_segs:
                self.segments.append(segment)
            else:
                self.segments[self.seg_idx] = segment
                self.seg_idx = (self.seg_idx + 1) % self.max_segs
        easy_tf_log.logkv('n_segments', len(self.segments))

    def sample_seg_pair(self):
        """
        Sample a random pair of segments which hasn't yet been tested.
        """
        segment_idxs = list(range(len(self.segments)))
        shuffle(segment_idxs)
        possible_pairs = combinations(segment_idxs, 2)
        for i1, i2 in possible_pairs:
            s1, s2 = self.segments[i1], self.segments[i2]
            if ((s1.hash, s2.hash) not in self.tested_pairs) and \
               ((s2.hash, s1.hash) not in self.tested_pairs):
                self.tested_pairs.add((s1.hash, s2.hash))
                self.tested_pairs.add((s2.hash, s1.hash))
                return s1, s2
        raise IndexError("No segment pairs yet untested")

    def ask_user(self, s1, s2):
        vid = []
        seg_len = len(s1)
        for t in range(seg_len):
            border = np.zeros((84, 10), dtype=np.uint8)
            # -1 => show only the most recent frame of the 4-frame stack
            frame = np.hstack((s1.frames[t][:, :, -1],
                               border,
                               s2.frames[t][:, :, -1]))
            vid.append(frame)
        n_pause_frames = 7
        vid.extend([vid[-1]] * n_pause_frames)
        self.vid_q.put(vid)

        while True:
            print("Segments {} and {}: ".format(s1.hash, s2.hash))
            choice = input()
            # L = "I prefer the left segment"
            # R = "I prefer the right segment"
            # E = "I don't have a clear preference between the two segments"
            # "" = "The segments are incomparable"
            if choice == "L" or choice == "R" or choice == "E" or choice == "":
                break
            else:
                print("Invalid choice '{}'".format(choice))

        if choice == "L":
            pref = (1.0, 0.0)
        elif choice == "R":
            pref = (0.0, 1.0)
        elif choice == "E":
            pref = (0.5, 0.5)
        elif choice == "":
            pref = None

        self.vid_q.put(VideoRenderer.pause_cmd)

        return pref
