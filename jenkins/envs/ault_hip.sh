#!/bin/bash

source $(dirname "$BASH_SOURCE")/ault.sh

module use /users/fthaler/public/jenkins/modules
module load gcc/8.3.0
module load cuda/10.1
module load hip-clang

export CXX=$(which hipcc)
export CC=$(which gcc)
export FC=$(which gfortran)

export GTRUN_BUILD_COMMAND='srun -w ault20 --time=01:00:00 make -j 64'
export GTRUN_SBATCH_NTASKS=1
export GTRUN_SBATCH_CPUS_PER_TASK=128
export GTRUN_SBATCH_MEM_BIND=local
export GTRUN_SBATCH_NODELIST='ault20'
export GTRUN_SBATCH_TIME='00:30:00'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG -march=znver1'

export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=64
export OMP_PLACES='{0}:64'

export CTEST_PARALLEL_LEVEL=1
