#!/bin/sh

source daint.sh

module load /users/vogtha/modules/compilers/clang/7.0.1

export CXX=$(which clang++)
export CC=$(which clang)

export GTCMAKE_GT_ENABLE_BACKEND_CUDA='OFF'
