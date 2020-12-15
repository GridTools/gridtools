#!/bin/bash

source $(dirname "$BASH_SOURCE")/ault.sh

# no module available for ROCm 3.9
export ROCM_PATH=/opt/rocm-3.9.0
export PATH="$PATH:$ROCM_PATH/bin"
# libraries are distributed over many dirs, so search them by name
rocm_lib_paths=$(find $ROCM_PATH -type d -name 'lib*' | tr '\n' ':')
export LIBRARY_PATH="$rocm_lib_paths$LIBRARY_PATH"
export LD_LIBRARY_PATH="$rocm_lib_paths$LD_LIBRARY_PATH"
module load gcc/10.1.0

export CXX=$(which hipcc)
export CC=$(which gcc)
export FC=$(which gfortran)

export GTRUN_BUILD_COMMAND='srun -p amdvega --time=03:00:00 make -j 64'
export GTRUN_SBATCH_NTASKS=1
export GTRUN_SBATCH_CPUS_PER_TASK=128
export GTRUN_SBATCH_MEM_BIND=local
export GTRUN_SBATCH_PARTITION=amdvega
export GTRUN_SBATCH_TIME='00:30:00'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -march=znver1'

export HIP_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=64
export OMP_PLACES='{0}:64'
export HCC_AMDGPU_TARGET=gfx906

export CTEST_PARALLEL_LEVEL=1
