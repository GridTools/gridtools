#!/bin/bash

if [ -n "${GRIDTOOLS_ROOT}" ]; then
    docker run --rm=true                                            \
               --volume=${GRIDTOOLS_ROOT}:/home/docker/gridtools    \
               --name=gridtools4py                                  \
               -P gridtools4py:cpu
else
    echo "Please set GRIDTOOLS_ROOT pointing to the source to be mounted into the container"
    exit 1
fi
