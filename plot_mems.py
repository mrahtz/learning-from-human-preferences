#!/usr/bin/env python

import argparse
import os.path as osp
from time import sleep

from pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()

logs = ['a2c', 'train', 'interface']
ion()
while True:
    clf()
    for i, log in enumerate(logs):
        with open(osp.join(args.dir, log + '.log')) as f:
            a2c = f.read().rstrip().split('\n')
        a2c_mems = [l.split()[1] for l in a2c]
        subplot(3, 1, i+1)
        title(log)
        plot(a2c_mems)
    pause(1.0)

show()
