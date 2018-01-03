#!/usr/bin/env python

import memory_profiler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('pid', type=int)
parser.add_argument('--log_name', default='mem.log')
args = parser.parse_args()

with open(args.log_name, 'w') as f:
    memory_profiler.memory_usage(
        proc=args.pid,
        stream=f,
        timeout=99999999,
        multiprocess=True)
