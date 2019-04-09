#!/bin/sh

source base.sh

module load daint-gpu
module load cudatoolkit/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52
module rm PrgEnv-cray
module rm CMake
module load /users/jenkins/easybuild/daint/haswell/modules/all/CMake/3.12.4

export BOOST_ROOT=$SCRATCH/../jenkins/install/boost/boost_1_67_0
export CUDATOOLKIT_HOME=$CUDA_PATH
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_G2G_PIPELINE=30
export CUDA_ARCH=sm_60

export GTCI_MPI_NODES=4
export GTCI_MPI_TASKS=4
export GTCI_QUEUE=normal
export GTCI_BUILD_THREADS=24
export GTCI_BUILD_COMMAND="srun -C gpu --account c14 -p cscsci --time=00:20:00"
