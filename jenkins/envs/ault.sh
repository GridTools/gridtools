#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

module load cuda
module load gcc/7.3.0
module load cmake/3.14.0

export BOOST_ROOT=/users/fthaler/checkouts/spack/opt/spack/linux-centos7-x86_64/gcc-8.3.0/boost-1.70.0-gk6agd5fm4a6t5x2ckzitv4lkw5ctasq
export CUDA_ARCH=sm_70

export GTRUN_BUILD_COMMAND='make -j 8'
export GTRUN_SBATCH_PARTITION='amdv100'
export GTRUN_SBATCH_NODES=1

export GTCMAKE_GT_ENABLE_BACKEND_CUDA=ON
export GTCMAKE_GT_ENABLE_BACKEND_X86=OFF
export GTCMAKE_GT_ENABLE_BACKEND_MC=OFF
export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=OFF
export GTCMAKE_GT_EXAMPLES_FORCE_CUDA=ON

export CUDA_AUTO_BOOST=0
export GCLOCK=1380

export CTEST_PARALLEL_LEVEL=1
