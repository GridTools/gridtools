#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module load /users/vogtha/modules/compilers/clang/7.0.1
module load gcc

export GTCMAKE_GT_CLANG_CUDA_MODE=NVCC-CUDA

export CXX=$(which clang++)
export CC=$(which clang)
export FC=$(which gfortran)

