#!/bin/bash

# TODO review all commented parts

source $(dirname "$BASH_SOURCE")/base.sh

module load cmake/3.14.5
module load craype-x86-skylake
module load craype-network-infiniband
module load slurm

export BOOST_ROOT=/project/c14/install/tsa/boost/boost_1_67_0/
#export CUDATOOLKIT_HOME=$CUDA_PATH
export CUDA_ARCH=sm_70

#export GTRUN_BUILD_COMMAND='srun -C gpu -p cscsci --time=00:20:00 make -j 24'
export GTRUN_BUILD_COMMAND='make -j 8'
export GTRUN_SBATCH_PARTITION='debug'
export GTRUN_SBATCH_NODES=1
export GTRUN_SBATCH_NTASKS_PER_CORE=1
export GTRUN_SBATCH_NTASKS_PER_NODE=1
#export GTRUN_SBATCH_CPUS_PER_TASK=24
export GTRUN_SBATCH_GRES='gpu:1'
export GTRUNMPI_SBATCH_PARTITION='debug'
export GTRUNMPI_SBATCH_NODES=4

#export CUDA_AUTO_BOOST=0
#export GCLOCK=1328
#export MPICH_RDMA_ENABLED_CUDA=1
#export MPICH_G2G_PIPELINE=30
#export OMP_NUM_THREADS=24
