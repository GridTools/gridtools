#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module load PrgEnv-gnu
module swap gcc/7.3.0

module load /project/csstaff/fthaler/install/daint/spack/share/spack/modules/cray-cnl6-haswell/llvm-9.0.0-gcc-7.3.0-3lmdwud

export CXX=$(which clang++)
export CC=$(which clang)
export FC=$(which gfortran)

export CXXFLAGS='--gcc-toolchain=/opt/gcc/7.3.0/snos/'
export CFLAGS='--gcc-toolchain=/opt/gcc/7.3.0/snos/'

export CTEST_PARALLEL_LEVEL=1

export GTCMAKE_GT_CUDA_COMPILATION_TYPE='Clang-CUDA'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export GTCMAKE_GT_ENABLE_BACKEND_CUDA=ON
export GTCMAKE_GT_ENABLE_BACKEND_X86=OFF
export GTCMAKE_GT_ENABLE_BACKEND_MC=OFF
export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=OFF
