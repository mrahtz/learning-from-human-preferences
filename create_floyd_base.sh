#!/bin/bash

set -o errexit

touch before_file

bash floyd_wrapper.sh

find / -type f -newer before_file | grep -v -e '^/proc' -e '^/sys' -e '^/output' | xargs -i cp --parents {} /output
