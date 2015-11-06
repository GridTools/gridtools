#!/bin/bash

if [ -f "examples/build_$1.sh" ]
then
    echo "build_$1.sh $2 $3 $4 $5 $6"
    if [ -d fusion ]
    then
        echo "fusion already there, replacing it"
	rm -rf fusion
        git clone https://github.com/ericniebler/fusion.git fusion
    else
        git clone https://github.com/ericniebler/fusion.git fusion
    fi

    pwd
    if [ -d "build" ]
    then
        rm -rf build
    fi
    mkdir -p build;
    cd build;

    eval "../examples/build_$1.sh $2 $3 $4 $5 $6"
else
    echo "ERROR: node $1 not supported"
fi
