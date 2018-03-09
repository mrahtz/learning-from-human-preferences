#!/usr/bin/env python
import queue
import time
from itertools import combinations
from multiprocessing import Process, Queue

import numpy as np

from scipy.ndimage import zoom

from utils import vid_proc

import logging


class PrefInterface:

    def __init__(self, headless, synthetic_prefs=False):
        self.vid_q = Queue()
        if not headless:
            Process(target=vid_proc, args=(self.vid_q,), daemon=True).start()
        self.synthetic_prefs = synthetic_prefs

    def recv_segments(self, segments, seg_pipe, max_segs):
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
            if len(segments) > max_segs:
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

    def run(self, seg_pipe, pref_pipe, max_segs):
        tested_pairs = []
        segments = []

        while True:
            self.recv_segments(segments, seg_pipe, max_segs)
            if len(segments) >= 2:
                break
            print("Preference interface waiting for segments")
            time.sleep(2.0)

        # TODO label segments
        while True:
            pair_idxs = []
            # If we've tested all the possible pairs of segments so far,
            # we might have to wait
            while len(pair_idxs) == 0:
                self.recv_segments(segments, seg_pipe, max_segs)
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
