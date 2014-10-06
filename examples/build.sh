#!/bin/bash

if [ -f "examples/build_$1.sh" ]
then
    echo \"/build_$1.sh\"
    git clone https://github.com/ericniebler/fusion.git ../fusion
    eval "./examples/build_$1.sh"
    rm -r ../fusion
else
    echo "ERROR: node $1 not supported"
fi