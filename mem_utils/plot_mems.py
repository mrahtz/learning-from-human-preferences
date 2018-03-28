#!/usr/bin/env python

"""
Plot process memory usage graphs recorded by utils.profile_memory.
"""

import argparse
from glob import glob
from os.path import join

from pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()


files = glob(join(args.dir, 'mem_*.log'))
for i, log in enumerate(files):
    with open(log) as f:
        lines = f.read().rstrip().split('\n')
    mems = [float(l.split()[1]) for l in lines]
    times = [float(l.split()[2]) for l in lines]
    rtimes = [t - times[0] for t in times]
    subplot(len(files), 1, i + 1)
    title(log)
    plot(rtimes, mems)

tight_layout()
show()
