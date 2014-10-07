#!/bin/bash

if [ -f "examples/build_$1.sh" ]
then
    echo "build_$1.sh $2 $3"
    if [ -d ../fusion ]
    then
	rm -rf ../fusion
    fi
    git clone https://github.com/ericniebler/fusion.git ../fusion

    if [ -d "build" ]
    then
        rm -rf "build"
    fi
    mkdir -p "build";
    cd "build";

    eval "../examples/build_$1.sh $2 $3"
else
    echo "ERROR: node $1 not supported"
fi
