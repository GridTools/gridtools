#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module load /users/vogtha/modules/compilers/clang/7.0.1
module load gcc

export GTCMAKE_GT_PREFER_CLANG_CUDA_OVER_NVCC_CUDA=OFF

export CXX=$(which clang++)
export CC=$(which clang)
export FC=$(which gfortran)

