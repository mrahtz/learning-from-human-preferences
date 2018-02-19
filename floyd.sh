#!/bin/bash

set -o errexit

find . | xargs touch

git clone https://github.com/openai/gym
cd gym
git reset --hard b5576dc23a5fcad0733042ab2ad440200ebb6209
pip install .[atari]
cd ..

cd gym-gridworld
pip install .
cd ..

pip install --upgrade cython cloudpickle

$*
