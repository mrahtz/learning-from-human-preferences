#!/bin/bash

if [[ $1 == "" ]]; then
    echo "Usage: $0 <directory>" >&2
    exit 1
fi
dir=$1

echo "Deleting:"
find $dir -size +5M
read -p "Proceed? " proceed
if [[ $proceed == 'y' ]]; then
    find $dir -size +5M | xargs rm
fi
