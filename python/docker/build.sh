#!/bin/bash

echo "This container image builds on top of the official NVIDIA CUDA container 'cuda:latest'"
echo "Press any key to continue or Ctrl+C to exit ..."
read -n 1

if [ -z "${GRIDTOOLS_ROOT}" ]; then
    echo "Please set GRIDTOOLS_ROOT pointing to the Gridtools source tree."
    exit 1
fi

#
# create an archive with the Gridtools source 
# before including it in the container
#
cd ${GRIDTOOLS_ROOT}
git archive -o ${GRIDTOOLS_ROOT}/python/docker/gridtools.tar.gz HEAD
cd -

docker build --rm -t gridtools4py .

rm -f ${GRIDTOOLS_ROOT}/python/docker/gridtools.tar.gz
