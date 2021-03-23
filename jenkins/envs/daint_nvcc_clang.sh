#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module switch cudatoolkit/10.2.89_3.29-7.0.2.1_3.5__g67354b4
module load /users/vogtha/modules/compilers/clang/7.0.1
module load gcc/8.3.0

export CXX=$(which clang++)
export CC=$(which clang)
export FC=$(which gfortran)
export CUDAHOSTCXX="$CXX"

export GTCMAKE_GT_CLANG_CUDA_MODE=NVCC-CUDA

export CTEST_PARALLEL_LEVEL=1
