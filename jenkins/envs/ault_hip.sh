#!/bin/bash

source $(dirname "$BASH_SOURCE")/ault.sh

module load rocm/3.8.0
# fix for broken module
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/rocm-3.8.0/llvm/lib"
module load gcc/10.1.0

export CXX=$(which hipcc)
export CC=$(which gcc)
export FC=$(which gfortran)

export GTRUN_BUILD_COMMAND='srun -w ault20 --time=03:00:00 make -j 64'
export GTRUN_SBATCH_NTASKS=1
export GTRUN_SBATCH_CPUS_PER_TASK=128
export GTRUN_SBATCH_MEM_BIND=local
export GTRUN_SBATCH_NODELIST=ault20
export GTRUN_SBATCH_TIME='00:30:00'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -march=znver1'

export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=64
export OMP_PLACES='{0}:64'
export HCC_AMDGPU_TARGET=gfx906

export CTEST_PARALLEL_LEVEL=1
