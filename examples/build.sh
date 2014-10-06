#!/bin/bash

if [ -f "examples/build_$1.sh" ]
then
    echo \"/build_$1.sh\"
    eval "./examples/build_$1.sh"
else
    echo "ERROR: node $1 not supported"
fi