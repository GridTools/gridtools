#!/bin/bash

source $(dirname "$BASH_SOURCE")/ault.sh

export MODULEPATH="$MODULEPATH:/users/fthaler/checkouts/amd-toolchain/module"
module load amd-toolchain

export CXX=$(which clang++)
export CC=$(which clang)
export FC=$(which gfortran)
export GTCMAKE_CMAKE_CUDA_HOST_COMPILER=$(which g++)
export GTCMAKE_GT_USE_CLANG_CUDA=ON
