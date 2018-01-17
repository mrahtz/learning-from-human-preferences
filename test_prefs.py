#!/usr/bin/env python3

import argparse
import queue
import time
from collections import OrderedDict
from multiprocessing import Process, Queue

import numpy as np

from dot_utils import predict_reward
from pref_interface import vid_proc
from reward_predictor import RewardPredictorEnsemble, batch_iter, load_pref_db
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true")
args = parser.parse_args()


def train(cmd_q):
    rp = RewardPredictorEnsemble(
        'train_reward', cluster_dict, load_network=args.load, dropout=0.0)
    stop = False
    while True:
        try:
            cmd_q.get(block=False)
            stop = True
            print("Stopping training...")
        except queue.Empty:
            pass

        if stop or args.load:
            time.sleep(1)
        else:
            rp.train(prefs_train, prefs_val, test_interval=1)


q = Queue()
Process(target=vid_proc, args=(q,)).start()

cluster_dict = {
    'test': ['localhost:2200'],
    'train_reward': ['localhost:2201'],
}

prefs_train = load_pref_db('train')
prefs_val = load_pref_db('val')

cmd_q = Queue()
p = Process(target=train, args=(cmd_q, ), daemon=True)
p.start()

rp = RewardPredictorEnsemble(
    'test', cluster_dict, load_network=args.load, dropout=0.0)

if args.load:
    cmd_q.put('stop')
else:
    time.sleep(60.0)
    cmd_q.put('stop')

print("Calculating preferences...")

predicted_prefs = []
for batch_n, batch in enumerate(batch_iter(prefs_train.prefs, batch_size=16)):
    print("Batch %d" % batch_n)
    s1s = []
    s2s = []
    for k1, k2, _ in batch:
        s1s.append(prefs_train.segments[k1])
        s2s.append(prefs_train.segments[k2])
    preds = rp.preferences(s1s, s2s, vote=True)
    predicted_prefs.extend(preds)

print("Test 1: are the network's preference outputs right?")

n_right = 0
n_wrong = 0
for i in range(len(prefs_train)):
    correct_pref = prefs_train.prefs[i][2]
    print(predicted_prefs[i], correct_pref)
    if tuple(predicted_prefs[i]) == correct_pref:
        n_right += 1
    else:
        n_wrong += 1
print(n_right, n_wrong)

input()

print("Calculating rewards...")
rewards = {}
for k, segment in prefs_train.segments.items():
    reward = np.sum(rp.reward_unnormalized(np.array(segment)))
    print("Segment %d: %.1f" % (k, reward))
    rewards[k] = reward
print("Calculating rewards done!")
print()

print("Test 2: are the preferences calculated based on the rewards right?")

n_right = 0
n_wrong = 0
for k1, k2, pref in prefs_train.prefs:
    print(k1, k2)
    r1 = rewards[k1]
    r2 = rewards[k2]
    if r1 > r2:
        pred_pref = (1.0, 0.0)
    elif r2 > r1:
        pred_pref = (0.0, 1.0)
    else:
        pred_pref = (0.5, 0.5)
    print(pred_pref, pref)
    if pred_pref == pref:
        n_right += 1
    else:
        n_wrong += 1
print(n_right, n_wrong)

input()

print("Rewards predicted for each segment:")
print(rewards)
print("Actual rewards:")
for seg in prefs_train.segments.values():
    print(predict_reward(seg))

input()

rewards_list = [(k, v) for k, v in rewards.items()]  # seg hash, reward
rewards_list.sort(key=lambda x: x[1])
samples = np.linspace(0, len(rewards), endpoint=False, num=5, dtype=int)
rewards_list = [x for i, x in enumerate(rewards_list) if i in samples]
print("Sample of segments sorted by rewards:")
print(rewards_list)
segments = np.array([prefs_train.segments[k] for k, _ in rewards_list])

vid = []
border = np.zeros((84, 10), dtype=np.uint8)
for t in range(len(segments[0])):
    frame = segments[0, t, :, :, 0]
    for n in range(1, len(segments)):
        f2 = segments[n, t, :, :, 0]
        frame = np.hstack((frame, border, f2))
    frame = zoom(frame, 2)
    vid.append(frame)
n_pause_frames = 7
vid.extend([vid[-1]] * n_pause_frames)
q.put(vid)
input()
