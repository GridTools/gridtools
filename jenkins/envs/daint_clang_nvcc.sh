#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)

export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'
export GTCMAKE_CMAKE_CUDA_HOST_COMPILER="$CXX"

export CUDAHOSTCXX="$CXX"
export CTEST_PARALLEL_LEVEL=1
export CXXFLAGS='-fno-cray-gpu -fno-cray'
export CFLAGS='-fno-cray-gpu -fno-cray-mallopt -fno-cray'
