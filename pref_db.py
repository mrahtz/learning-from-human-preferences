import pickle
import queue
from os import path as osp

import numpy as np


class Segment:
    def __init__(self):
        self.frames = []
        self.rewards = []
        self.hash = None

    def append(self, frame, reward):
        self.frames.append(frame)
        self.rewards.append(reward)

    def finalise(self, seg_id=None):
        if seg_id is not None:
            self.hash = seg_id
        else:
            # How expensive is this? About 0.5 ms.
            self.hash = hash(np.array(self.frames).tostring())

    def __len__(self):
        return len(self.frames)


class PrefDB:
    def __init__(self):
        self.segments = {}
        self.seg_refs = {}
        self.prefs = []

    def append(self, s1, s2, mu):
        k1 = hash(np.array(s1).tostring())
        k2 = hash(np.array(s2).tostring())

        for k, s in zip([k1, k2], [s1, s2]):
            if k not in self.segments.keys():
                self.segments[k] = s
                self.seg_refs[k] = 1
            else:
                self.seg_refs[k] += 1

        tup = (k1, k2, mu)
        self.prefs.append(tup)

    def del_first(self):
        self.del_pref(0)

    def del_pref(self, n):
        if n >= len(self.prefs):
            raise IndexError("Preference {} doesn't exist".format(n))
        k1, k2, _ = self.prefs[n]
        for k in [k1, k2]:
            if self.seg_refs[k] == 1:
                del self.segments[k]
                del self.seg_refs[k]
            else:
                self.seg_refs[k] -= 1
        del self.prefs[n]

    def __len__(self):
        return len(self.prefs)


def save_pref_db(pref_db, fname):
    with open(fname, 'wb') as pkl_file:
        pickle.dump(pref_db, pkl_file)


def load_pref_db(pref_dir):
    train_fname = osp.join(pref_dir, 'train_initial.pkl')
    with open(train_fname, 'rb') as pkl_file:
        pref_db_train = pickle.load(pkl_file)
        print("Loaded training preferences from '{}'".format(train_fname))

    val_fname = osp.join(pref_dir, 'val_initial.pkl')
    with open(val_fname, 'rb') as pkl_file:
        pref_db_val = pickle.load(pkl_file)
        print("Loaded validation preferences from '{}'".format(val_fname))

    return pref_db_train, pref_db_val


def save_prefs(pref_db_train, pref_db_val, save_dir, name):
    fname = osp.join(save_dir, "train_{}.pkl".format(name))
    save_pref_db(pref_db_train, fname)
    fname = osp.join(save_dir, "val_{}.pkl".format(name))
    save_pref_db(pref_db_val, fname)


def recv_prefs(pref_pipe, pref_db_train, pref_db_val, max_prefs):
    val_fraction = 0.2
    while True:
        try:
            s1, s2, mu = pref_pipe.get(timeout=0.1)
        except queue.Empty:
            break

        if np.random.rand() < val_fraction:
            pref_db_val.append(s1, s2, mu)
        else:
            pref_db_train.append(s1, s2, mu)

        if len(pref_db_val) > max_prefs * val_fraction:
            pref_db_val.del_first()
        assert len(pref_db_val) <= max_prefs * val_fraction

        if len(pref_db_train) > max_prefs * (1 - val_fraction):
            pref_db_train.del_first()
        assert len(pref_db_train) <= max_prefs * (1 - val_fraction)
