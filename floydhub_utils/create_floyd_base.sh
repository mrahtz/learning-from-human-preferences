#!/bin/bash

# Create a FloydHub dataset consisting of files changed by the setup wrapper
# (e.g. packages dependencies) for quicker launching of jobs

set -o errexit

touch before_file

bash floyd_wrapper.sh

echo "Copying changed files..."
find / -type f -newer before_file | grep -v -e '^/proc' -e '^/sys' -e '^/output' -e '^/code' -e '^/floydlocaldata' -e '^/root' | xargs -i cp --parents {} /output
echo "Done!"
