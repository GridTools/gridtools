#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module load /users/vogtha/modules/compilers/clang/7.0.1
module load gcc

export CXX=$(which clang++)
export CC=$(which clang)
export FC=$(which gfortran)

export GTCMAKE_GT_ENABLE_BACKEND_CUDA='OFF'
