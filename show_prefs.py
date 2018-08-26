#!/usr/bin/env python3
"""
Display examples of the specified preference database
(with the less-preferred segment on the left,
and the more-preferred segment on the right)
(skipping over equally-preferred segments)
"""

import argparse
import gzip
import pickle
from multiprocessing import Queue

import numpy as np

from utils import VideoRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefs", help=".pkl.gz file")
    args = parser.parse_args()

    with gzip.open(args.prefs, 'rb') as pkl_file:
        print("Loading preferences from '{}'...".format(args.prefs), end="")
        prefs = pickle.load(pkl_file)
        print("done!")

    print("{} preferences found".format(len(prefs)))
    print("Preferred segment on the right")

    q = Queue()
    VideoRenderer(q, zoom=2, mode=VideoRenderer.restart_on_get_mode)

    for k1, k2, pref in prefs.prefs:
        if pref == (0.5, 0.5):
            continue

        if pref == (0.0, 1.0):
            s1 = np.array(prefs.segments[k1])
            s2 = np.array(prefs.segments[k2])
        elif pref == (1.0, 0.0):
            s1 = np.array(prefs.segments[k2])
            s2 = np.array(prefs.segments[k1])
        else:
            raise Exception("Unexpected preference", pref)

        vid = []
        border = np.ones((84, 10), dtype=np.uint8) * 128
        for t in range(len(s1)):
            # -1 => select the last frame in the 4-frame stack
            f1 = s1[t, :, :, -1]
            f2 = s2[t, :, :, -1]
            frame = np.hstack((f1, border, f2))
            vid.append(frame)
        n_pause_frames = 10
        for _ in range(n_pause_frames):
            vid.append(np.copy(vid[-1]))
        q.put(vid)
        input()


if __name__ == '__main__':
    main()
