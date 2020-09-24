#!/bin/bash

source $(dirname "$BASH_SOURCE")/tsa.sh

module load PrgEnv-gnu/19.2

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)
export CUDAHOSTCXX="$CXX"

export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'
export GTCMAKE_GT_REQUIRE_OpenMP="ON"
export GTCMAKE_GT_REQUIRE_GPU="ON"

export CTEST_PARALLEL_LEVEL=1
