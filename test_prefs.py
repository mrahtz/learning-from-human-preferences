#!/usr/bin/env python3

import argparse
import os
from multiprocessing import Process, Queue

from pref_interface import vid_proc
q = Queue()
Process(target=vid_proc, args=(q,)).start()

import numpy as np
import params
from dot_utils import predict_reward
from reward_predictor import RewardPredictorEnsemble, batch_iter, load_pref_db
from scipy.ndimage import zoom

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def ps(cluster_dict, rp_ckpt_dir):
    RewardPredictorEnsemble(
        name='ps',
        cluster_dict=cluster_dict,
        load_network=True,
        rp_ckpt_dir=rp_ckpt_dir)


parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("prefs_dir")
args = parser.parse_args()

params.init_params()
params.params['network'] = 'onelayer'
params.params['debug'] = False

cluster_dict = {
    'ps': ['localhost:2200'],
    'train_reward': ['localhost:2201']
}
Process(target=ps, args=(cluster_dict, args.ckpt), daemon=True).start()
rp = RewardPredictorEnsemble(
    name='train_reward',
    cluster_dict=cluster_dict,
    load_network=True,
    rp_ckpt_dir=args.ckpt)

prefs_train, prefs_val = load_pref_db(args.prefs_dir)

print("Calculating preferences...")

while len(prefs_train) > 8:
    prefs_train.del_first()

predicted_prefs = []
for batch_n, batch in enumerate(batch_iter(prefs_train.prefs, batch_size=64)):
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
for i, (k, segment) in enumerate(prefs_train.segments.items()):
    reward = np.sum(rp.reward(np.array(segment)))
    print("Segment %d/%d (key %d): %.1f" % (i + 1, len(prefs_train.segments), k,
                                            reward))
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
