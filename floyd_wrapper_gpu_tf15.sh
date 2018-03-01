#!/bin/bash

set -o errexit

cp -r /gpu_tf15/* /
$*
