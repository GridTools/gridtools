#!/bin/bash

if [ -z "${GRIDTOOLS_ROOT}" ]; then
    echo "GRIDTOOLS_ROOT not found ... the container's source code will be used"
    docker run --rm=true                                            \
               --name=gridtools4py                                  \
               -P gridtools4py:cpu
else
    echo "Mounting <${GRIDTOOLS_ROOT}> under container's </home/docker/gridtools> ..."
    docker run --rm=true                                            \
               --volume=${GRIDTOOLS_ROOT}:/home/docker/gridtools    \
               --name=gridtools4py                                  \
               -P gridtools4py:cpu
fi
