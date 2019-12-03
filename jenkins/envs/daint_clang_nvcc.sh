#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)

export CTEST_PARALLEL_LEVEL=1

export CXXFLAGS='-fno-cray-gpu -fno-cray-mallopt -fno-cray'
export CFLAGS='-fno-cray-gpu -fno-cray-mallopt -fno-cray'

export GTCMAKE_GT_CUDA_COMPILATION_TYPE='Clang-CUDA'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export GTCMAKE_GT_ENABLE_BACKEND_CUDA=ON
export GTCMAKE_GT_ENABLE_BACKEND_X86=OFF
export GTCMAKE_GT_ENABLE_BACKEND_MC=OFF
export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=OFF
export GTCMAKE_GT_USE_MPI=OFF
