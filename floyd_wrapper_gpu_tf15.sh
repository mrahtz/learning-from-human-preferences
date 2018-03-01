#!/bin/bash

set -o errexit

echo "Copying base files..."
cp -r /gpu_tf15/* /
echo "done!"

$*
