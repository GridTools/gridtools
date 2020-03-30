#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/7.3.0

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)

export GTCMAKE_CMAKE_CUDA_HOST_COMPILER="$CXX"
export GTCMAKE_CMAKE_CXX_FLAGS='-march=haswell'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export CUDAHOSTCXX="$CXX"
export CTEST_PARALLEL_LEVEL=1

