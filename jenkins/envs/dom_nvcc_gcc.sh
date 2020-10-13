#!/bin/bash

source $(dirname "$BASH_SOURCE")/dom.sh

module switch cudatoolkit cudatoolkit/11.0.2_3.33-7.0.2.1_3.1__g1ba0366

module swap PrgEnv-cray PrgEnv-gnu

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)
export CUDAHOSTCXX="$CXX"

export GTCMAKE_CMAKE_CXX_FLAGS='-march=haswell'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export CTEST_PARALLEL_LEVEL=1
