#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module load gcc/7.3.0

export CXX=$(which clang++)
export CC=$(which clang)
export FC=$(which gfortran)

export CTEST_PARALLEL_LEVEL=1

export GTCMAKE_GT_CUDA_COMPILATION_TYPE='Clang-CUDA'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export GTCMAKE_GT_ENABLE_BACKEND_CUDA=ON
export GTCMAKE_GT_ENABLE_BACKEND_X86=OFF
export GTCMAKE_GT_ENABLE_BACKEND_MC=OFF
export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=OFF
