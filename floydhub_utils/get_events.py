#!/usr/bin/env python3

"""
Download all TensorFlow event files from the specified jobs' output files.
"""

import argparse
import os
import os.path as osp
import subprocess
import time
from multiprocessing import Process


def get(job_id, out_dir):
    print("Listing files...")
    cmd = "floyd data listfiles {}/output".format(job_id)
    files = subprocess.check_output(cmd.split()).decode().split('\n')
    event_files = [f for f in files if 'events.out.tfevents' in f]
    download_dir = osp.join(out_dir, job_id)
    os.makedirs(download_dir, exist_ok=True)
    for event_file in event_files:
        print("Downloading {}...".format(event_file))
        cmd = "floyd data getfile {}/output {}".format(job_id, event_file)
        subprocess.call(cmd.split())

        path = os.path.dirname(event_file)
        fname = os.path.basename(event_file)
        full_dir = osp.join(download_dir, path)
        os.makedirs(full_dir, exist_ok=True)
        os.rename(fname, osp.join(full_dir, fname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("job_ids", nargs='*')
    args = parser.parse_args()

    for job_id in args.job_ids:
        Process(target=get, args=(job_id, args.dir)).start()
        time.sleep(0.5)


if __name__ == '__main__':
    main()
