#!/bin/sh

source kesch.sh

export G2G=2
export MV2_USE_GPUDIRECT=0
export MV2_USE_RDMA_FAST_PATH=0

export CTEST_PARALLEL_LEVEL=1

export GTCMAKE_GT_ENABLE_BACKEND_CUDA=ON
