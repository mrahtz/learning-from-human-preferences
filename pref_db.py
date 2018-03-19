import gzip
import pickle

import numpy as np


class Segment:
    """
    A short recording of agent's behaviour in the environment,
    consisting of a number of video frames and the rewards it received
    during those frames.
    """

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
            # This looks expensive, but don't worry -
            # it only takes about 0.5 ms.
            self.hash = hash(np.array(self.frames).tostring())

    def __len__(self):
        return len(self.frames)


class PrefDB:
    """
    A circular database of preferences about pairs of segments.

    For each preference, we store the preference itself
    (mu in the paper) and the two segments the preference refers to.
    Segments are stored with deduplication - so that if multiple
    preferences refer to the same segment, the segment is only stored once.
    """

    def __init__(self, maxlen):
        self.segments = {}
        self.seg_refs = {}
        self.prefs = []
        self.maxlen = maxlen

    def append(self, s1, s2, pref):
        k1 = hash(np.array(s1).tostring())
        k2 = hash(np.array(s2).tostring())

        for k, s in zip([k1, k2], [s1, s2]):
            if k not in self.segments.keys():
                self.segments[k] = s
                self.seg_refs[k] = 1
            else:
                self.seg_refs[k] += 1

        tup = (k1, k2, pref)
        self.prefs.append(tup)

        if len(self.prefs) > self.maxlen:
            self.del_first()

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

    def save(self, path):
        with gzip.open(path, 'wb') as pkl_file:
            pickle.dump(self, pkl_file)

    @staticmethod
    def load(path):
        with gzip.open(path, 'rb') as pkl_file:
            pref_db = pickle.load(pkl_file)
        return pref_db