#!/usr/bin/env python3

import pickle
from itertools import combinations
from multiprocessing import Process, Queue

import numpy as np

import gym
import gym_gridworld
from baselines.common.atari_wrappers import wrap_deepmind
from dot_utils import predict_preference
from pref_interface import vid_proc
from scipy.ndimage import zoom
from utils import PrefDB

show = False
n_segments = 10
n_frames = 25
segments = []
prefs = []


def update_obs(obs, raw_obs, nc):
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = raw_obs
    return obs


env = wrap_deepmind(gym.make("GridWorldNoFrameskip-v4"))

nstack = 4
nenvs = 1
nh, nw, _ = env.observation_space.shape
obs = np.zeros((nenvs, nh, nw, nstack), dtype=np.uint8)

w, h, _ = env.unwrapped.observation_space.shape

# Generate segments

print("Generating segments...")
for seg_n in range(n_segments):
    segment = []
    x = np.random.randint(low=0, high=w)
    y = np.random.randint(low=0, high=h)

    obs = np.zeros((nenvs, nh, nw, nstack), dtype=np.uint8)
    env.unwrapped.pos = [x, y]

    for i in range(4):
        action = env.action_space.sample()
        raw_obs, _, _, _ = env.step(action)
        obs = update_obs(obs, raw_obs, nc=1)

    for _ in range(n_frames):
        action = env.action_space.sample()
        raw_obs, _, _, _ = env.step(action)
        obs = update_obs(obs, raw_obs, nc=1)
        segment.append(obs[0])
    segments.append(segment)

segments = np.array(segments)

if show:
    q = Queue()
    Process(target=vid_proc, args=(q, )).start()

# Generate preferences for segments


idxs = range(n_segments)
pair_idxs = list(combinations(idxs, 2))
np.random.shuffle(pair_idxs)

print("Generating %d preferences..." % len(pair_idxs))

p_train = PrefDB()
p_val = PrefDB()

for i1, i2 in pair_idxs:
    s1 = segments[i1]
    s2 = segments[i2]
    pref = predict_preference(s1, s2)

    if np.random.rand() < 0.2:
        p_val.append(s1, s2, pref)
    else:
        p_train.append(s1, s2, pref)

    if show:
        vid = []
        border = np.zeros((84, 10), dtype=np.uint8)
        for t in range(n_frames):
            frame = np.hstack((s1[t, :, :, 0], border, s2[t, :, :, 0]))
            frame = zoom(frame, 2)
            vid.append(frame)
        n_pause_frames = 7
        vid.extend([vid[-1]] * n_pause_frames)
        q.put(vid)
        input()

print("Saving preferences...")

with open('pref_db_train.pkl', 'wb') as pkl_file:
    pickle.dump(p_train, pkl_file)
with open('pref_db_val.pkl', 'wb') as pkl_file:
    pickle.dump(p_val, pkl_file)
