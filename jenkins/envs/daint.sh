#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

module load daint-gpu
module load cudatoolkit/10.1.105_3.27-7.0.1.1_4.1__ga311ce7
module rm CMake
module load /users/jenkins/easybuild/daint/haswell/modules/all/CMake/3.14.5

export BOOST_ROOT=$SCRATCH/../jenkins/install/boost/boost_1_67_0
export CUDATOOLKIT_HOME=$CUDA_PATH
export CUDA_ARCH=sm_60

export GTRUN_BUILD_COMMAND='srun -C gpu -p cscsci --time=00:20:00 make -j 24'
export GTRUN_SBATCH_PARTITION='cscsci'
export GTRUN_SBATCH_NODES=1
export GTRUN_SBATCH_NTASKS_PER_CORE=2
export GTRUN_SBATCH_NTASKS_PER_NODE=1
export GTRUN_SBATCH_CPUS_PER_TASK=24
export GTRUN_SBATCH_CONSTRAINT='gpu'
export GTRUNMPI_SBATCH_PARTITION='normal'
export GTRUNMPI_SBATCH_NODES=4

export CUDA_AUTO_BOOST=0
export GCLOCK=1328
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_G2G_PIPELINE=30
export OMP_NUM_THREADS=24
