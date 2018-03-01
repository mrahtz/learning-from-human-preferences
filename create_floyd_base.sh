#!/bin/bash

set -o errexit

touch before_file

bash floyd_wrapper.sh

echo "Copying changes files..."
find / -type f -newer before_file | grep -v -e '^/proc' -e '^/sys' -e '^/output' -e '^/code' -e '^/floydlocaldata' -e '^/root' | xargs -i cp --parents {} /output
echo "Done!"
