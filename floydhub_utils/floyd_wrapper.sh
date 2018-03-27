#!/bin/bash

# Install dependencies, then run whatever was specified in the arguments

set -o errexit

# By default, FloydHub copies files with a timestamp of 0 seconds since epoch,
# which breaks pip sometimes
find . | xargs touch

pip install pipenv
# --site-packages so that we pick up the system TensorFlow
pipenv --site-packages install

pipenv run $*
