#!/bin/bash

# Copy files from base dataset (created by create_floyd_base.sh),
# then run whatever was specified in the arguments

set -o errexit

echo "Copying base files..."
cp -r /base_files/* /
echo "done!"

$*
