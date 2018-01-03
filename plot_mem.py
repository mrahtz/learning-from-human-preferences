#!/usr/bin/env python

from pylab import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("log_file")
args = parser.parse_args()

with open(args.log_file) as f:
    lines = f.read().rstrip().split('\n')
    
mems = {}
for line in lines:
    fields = line.split(' ')
    key = fields[0]
    if key == 'MEM':
        mem = key
        usage = fields[1]
    elif key == 'CHLD':
        mem = fields[1]
        usage = fields[2]
    else:
        raise Exception(key)
    if not mem in mems:
        mems[mem] = []
    mems[mem].append(usage)

n = len(mems)

for i, (mem, l) in enumerate(sorted(mems.items())):
    subplot(n, 1, i+1)
    title(mem)
    plot(l)

show()
