#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module load /users/vogtha/modules/compilers/clang/7.0.1

export CXX=$(which clang++)
export CC=$(which clang)
export GTCMAKE_CMAKE_CUDA_HOST_COMPILER="$CXX"

export CTEST_PARALLEL_LEVEL=1

export GTCMAKE_GT_ENABLE_BACKEND_CUDA=ON
export GTCMAKE_GT_ENABLE_BACKEND_X86=OFF
export GTCMAKE_GT_ENABLE_BACKEND_MC=OFF
export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=OFF
export GTCMAKE_GT_EXAMPLES_FORCE_CUDA=ON
