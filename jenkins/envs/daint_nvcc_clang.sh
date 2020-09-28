#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module load /users/vogtha/modules/compilers/clang/7.0.1
module load gcc

export CXX=$(which clang++)
export CC=$(which clang)
export FC=$(which gfortran)
export CUDAHOSTCXX="$CXX"

export GTCMAKE_GT_CLANG_CUDA_MODE=NVCC-CUDA

export CTEST_PARALLEL_LEVEL=1
