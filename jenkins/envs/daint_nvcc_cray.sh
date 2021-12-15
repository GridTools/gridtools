#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint_cray.sh

export GTCMAKE_GT_CLANG_CUDA_MODE=NVCC-CUDA

export CTEST_PARALLEL_LEVEL=1
