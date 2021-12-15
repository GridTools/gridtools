#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint_cray.sh

module switch cudatoolkit/11.0.2_3.38-8.1__g5b73779 cudatoolkit/11.2.0_3.39-2.1__gf93aa1c

export GTCMAKE_GT_CLANG_CUDA_MODE=NVCC-CUDA

export CTEST_PARALLEL_LEVEL=1

