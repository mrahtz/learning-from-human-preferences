#!/usr/bin/env python3

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
        subprocess.check_output(cmd.split())

        path = '/'.join(event_file[1:].split('/')[:-1])
        fname = event_file.split('/')[-1]

        full_dir = osp.join(download_dir, path)
        os.makedirs(full_dir, exist_ok=True)
        cmd = "mv {} {}".format(fname, full_dir)
        subprocess.check_output(cmd.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("ids", nargs='*')
    args = parser.parse_args()

    for job_id in args.ids:
        Process(target=get, args=(job_id, args.dir)).start()
        time.sleep(0.5)


if __name__ == '__main__':
    main()
