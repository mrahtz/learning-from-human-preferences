#!/usr/bin/env python3

import argparse
import pickle
from functools import cmp_to_key
from multiprocessing import Process, Queue

import numpy as np

from pref_interface import vid_proc
from scipy.ndimage import zoom

q = Queue()
Process(target=vid_proc, args=(q, ), daemon=True).start()

parser = argparse.ArgumentParser()
parser.add_argument("prefs")
args = parser.parse_args()

with open(args.prefs, 'rb') as pkl_file:
    print("Loading preferences from '{}'...".format(args.prefs), end="")
    prefs = pickle.load(pkl_file)
    print("done!")

prefs.prefs_dict = {}
for k1, k2, mu in prefs.prefs:
    prefs.prefs_dict[(k1, k2)] = mu


def cmpf(k1, k2):
    try:
        pref = prefs.get_pref(k1, k2)
    except KeyError:
        return 0

    if pref == (0.0, 1.0):
        return -1
    elif pref == (1.0, 0.0):
        return +1
    else:
        return 0


keys = list(prefs.segments.keys())
keys.sort(key=cmp_to_key(cmpf))
for k1, k2 in zip(keys[:-1], keys[1:]):
    try:
        prefs.get_pref(k1, k2)
        print(prefs.prefs_dict[(k1, k2)])
    except KeyError:
        print("?")

segments = [
    prefs.segments[keys[i]]
    for i in np.linspace(0, len(keys), endpoint=False, num=5, dtype=int)
]
segments = np.array(segments)

vid = []
border = np.ones((84, 10), dtype=np.uint8) * 128
for t in range(len(segments[0])):
    # Start with a frame of the leftmost video
    frame = segments[0, t, :, :, 0]
    for n in range(1, len(segments)):
        # Stack frames from other videos to the right
        f2 = segments[n, t, :, :, 0]
        frame = np.hstack((frame, border, f2))
    frame = zoom(frame, 2)
    vid.append(frame)
n_pause_frames = 10
vid.extend([vid[-1]] * n_pause_frames)
q.put(vid)
input()
