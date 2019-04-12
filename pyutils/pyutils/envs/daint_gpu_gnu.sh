#!/bin/sh

source daint.sh

module load PrgEnv-gnu
module swap gcc/7.3.0

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)
export GTCMAKE_CMAKE_CUDA_HOST_COMPILER="$CXX"
