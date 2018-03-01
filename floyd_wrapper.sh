#!/bin/bash

set -o errexit

find . | xargs touch

git clone https://github.com/openai/gym
cd gym
git reset --hard b5576dc23a5fcad0733042ab2ad440200ebb6209
pip install .[atari]
cd ..

git clone https://github.com/mrahtz/gym-moving-dot
cd gym-moving-dot
pip install .
cd ..

git clone https://github.com/mrahtz/easy-tf-log
cd easy-tf-log
pip install .
cd ..

pip install --upgrade cython cloudpickle joblib azure==1.0.3 mpi4py

$*
