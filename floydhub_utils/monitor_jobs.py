#!/usr/bin/env python

"""
Monitor running FloydHub jobs and send a macOS notification whenever
one finishes
"""

import re
import subprocess
import time
import sys


def display_notification(title, text):
    osa_cmd = 'display notification "{}" with title "{}"'.format(text, title)
    subprocess.call(['osascript', '-e', osa_cmd])


def main():
    running_jobs = set()
    while True:
        try:
            output = subprocess.check_output(['floyd', 'status'])
            output = output.decode().strip()
        except subprocess.CalledProcessError:
            continue
        output = output.split('\n')
        output = output[2:]  # Skip header lines

        for line in output:
            # Consider two or more spaces the field separator
            fields = re.sub(r"   *", '\t', line).split('\t')
            job_name = fields[0]
            job_id = job_name.split('/')[-1]
            status = fields[2]

            if status == 'running' and job_id not in running_jobs:
                print("Found new running job {}".format(job_id))
                running_jobs.add(job_id)
            elif ((status == 'shutdown' or status == 'success')
                  and job_id in running_jobs):
                print("Job {} finished".format(job_id))
                running_jobs.remove(job_id)
                display_notification("FloydHub job finished",
                                     "Job {} finished".format(job_id))

        time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
