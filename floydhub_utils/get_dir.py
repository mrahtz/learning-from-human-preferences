#!/usr/bin/env python3

"""
Download the specified directory from the specified job's output files.
"""

import argparse
import os
import os.path as osp
import subprocess
import time
from multiprocessing import Process


def getfile(f):
    dirname = osp.dirname(f)
    os.makedirs(dirname, exist_ok=True)
    print("Downloading {}...".format(f))
    cmd = "floyd data getfile {}/output {}".format(args.job_id, f)
    subprocess.check_output(cmd.split())
    os.rename(osp.basename(f), osp.join(dirname, osp.basename(f)))


parser = argparse.ArgumentParser()
parser.add_argument("job_id")
parser.add_argument("dir")
args = parser.parse_args()

print("Listing files...")
cmd = "floyd data listfiles {}/output".format(args.job_id)
allfiles = subprocess.check_output(cmd.split()).decode().split('\n')
dirfiles = [f for f in allfiles if f.startswith(args.dir + '/')]
dirfiles = [f for f in dirfiles if not f.endswith('/')]

for f in dirfiles:
    Process(target=getfile, args=(f, )).start()
    time.sleep(0.5)
