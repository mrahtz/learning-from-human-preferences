#!/bin/bash

set -o errexit
echo "Copying files..."
cp -r /python/* /usr/local/lib/python3.6/
echo "Done!"
echo
$*
