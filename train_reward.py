#!/usr/bin/env python3

from multiprocessing import Queue
from reward_predictor import RewardPredictorEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true')
args = parser.parse_args()

cluster_dict = {
    'train_reward': ['localhost:2201'],
}

rp = RewardPredictorEnsemble('train_reward', cluster_dict, n_preds=3, load=args.load)
cmd_q = Queue()
q = Queue()
rp.train(q, load=True, cmd_pipe=cmd_q)
