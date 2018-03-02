#!/bin/bash

set -o errexit

echo "Copying base files..."
cp -r /cpu_tf15/* /
echo "done!"

$*
