#!/bin/bash

source $(dirname "$BASH_SOURCE")/tsa.sh

module load PrgEnv-gnu/19.2

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)

export GTCMAKE_CMAKE_CUDA_HOST_COMPILER="$CXX"
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export CUDAHOSTCXX="$CXX"
export CTEST_PARALLEL_LEVEL=1
