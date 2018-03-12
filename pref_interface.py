#!/usr/bin/env python
import queue
import time
from itertools import combinations
from multiprocessing import Process, Queue
import easy_tf_log

import numpy as np

from scipy.ndimage import zoom

from utils import vid_proc

import logging


def sample_seg_pair(segments, tested_pairs):
    """
    Sample a pair of segments which hasn't yet been tested.
    """
    possible_pairs = list(combinations(range(len(segments)), 2))
    # Get a random pair; see if it's already been tested;
    # if it hasn't, return it; else, remove that pair from the list,
    # and try again.
    while True:
        idx = np.random.choice(range(len(possible_pairs)))
        i1, i2 = possible_pairs[idx]
        s1, s2 = segments[i1], segments[i2]
        if ((s1.hash, s2.hash) not in tested_pairs) and \
           ((s2.hash, s1.hash) not in tested_pairs):
            return s1, s2
        elif len(possible_pairs) > 1:
            if idx == len(possible_pairs) - 1:
                possible_pairs.pop()
            else:
                possible_pairs[idx] = possible_pairs.pop()
        else:
            raise IndexError("No segment pairs yet untested")


class PrefInterface:

    def __init__(self, headless, synthetic_prefs=False):
        self.vid_q = Queue()
        if not headless:
            Process(target=vid_proc, args=(self.vid_q,), daemon=True).start()
        self.synthetic_prefs = synthetic_prefs

    def recv_segments(self, segments, seg_pipe, max_segs):
        while True:
            try:
                segment = seg_pipe.get(timeout=0.1)
            except queue.Empty:
                break
            segments.append(segment)
            if len(segments) > max_segs:
                # O(1) removal of last element :)
                segments[0] = segments.pop()
        easy_tf_log.logkv('n_segments', len(segments))

    def ask_user(self, s1, s2):
        border = np.zeros((84, 10), dtype=np.uint8)

        seg_len = len(s1)
        vid = []
        for t in range(seg_len):
            # Show only the most recent frame of the 4-frame stack
            frame = np.hstack((s1.frames[t][:, :, -1], border, s2.frames[t][:, :, -1]))
            frame = zoom(frame, 4)
            vid.append(frame)
        n_pause_frames = 7
        vid.extend([vid[-1]] * n_pause_frames)
        self.vid_q.put(vid)

        while True:
            print("Segments {} and {}: ".format(s1.hash, s2.hash))
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
        tested_pairs = set()
        segments = []

        while True:
            self.recv_segments(segments, seg_pipe, max_segs)
            if len(segments) >= 2:
                break
            print("Preference interface waiting for segments")
            time.sleep(2.0)

        # TODO label segments
        while True:
            # If we've tested all the possible pairs of segments so far,
            # we might have to wait
            while True:
                self.recv_segments(segments, seg_pipe, max_segs)
                try:
                    s1, s2 = sample_seg_pair(segments,
                                             tested_pairs=tested_pairs)
                except IndexError:
                    continue
                else:
                    break

            logging.debug("Querying preference for segments", s1.hash, "and", s2.hash)

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
                # We don't need the rewards from this point on
                pref_pipe.put((s1.frames, s2.frames, pref))

            tested_pairs.add((s1.hash, s2.hash))
            tested_pairs.add((s2.hash, s1.hash))
