#!/bin/bash

if [ -f "examples/build_$1.sh" ]
then
    echo "build_$1.sh $2 $3"
    if [ -d ../fusion ]
    then
	rm -rf ../fusion
    fi
    git clone https://github.com/ericniebler/fusion.git ../fusion

    if [ -d "build/$1/$2/$3" ]
    then
        rm -rf "build/$1/$2/$3"
    fi
    mkdir -p "build/$1/$2/$3";
    cd "build/$1/$2/$3";

    eval "../../../../examples/build_$1.sh $2 $3"
else
    echo "ERROR: node $1 not supported"
fi
