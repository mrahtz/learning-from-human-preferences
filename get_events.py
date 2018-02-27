#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("ids", nargs='*')
args = parser.parse_args()

for id in args.ids:
    print("Listing files...")
    cmd = "floyd data listfiles {}/output".format(id)
    files = subprocess.check_output(cmd.split()).decode().split('\n')
    event_files = [f for f in files if 'events.out.tfevents' in f]
    download_dir = osp.join(args.dir, id)
    os.makedirs(download_dir, exist_ok=True)
    for event_file in event_files:
        print("Downloading {}...".format(event_file))
        cmd = "floyd data getfile {}/output {}".format(id, event_file)
        subprocess.check_output(cmd.split())

        path = '/'.join(event_file[1:].split('/')[:-1])
        fname = event_file.split('/')[-1]

        dir = osp.join(download_dir, path)
        os.makedirs(dir, exist_ok=True)
        cmd = "mv {} {}".format(fname, dir)
        subprocess.check_output(cmd.split())
