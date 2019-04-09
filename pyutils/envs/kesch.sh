#!/bin/sh

module load craype-network-infiniband
module load craype-haswell
module load craype-accel-nvidia35
module load cray-libsci
module load cudatoolkit/8.0.61
module load mvapich2gdr_gnu/2.2_cuda_8.0
module load gcc/5.4.0-2.26
module load /users/jenkins/easybuild/kesch/modules/all/cmake/3.12.4

export BOOST_ROOT=/project/c14/install/kesch/boost/boost_1_67_0
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export CUDATOOLKIT_HOME=$CUDA_PATH
export CUDA_ARCH=sm_37
export LD_PRELOAD=/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5/lib64/libmpi.so
export CUDA_AUTO_BOOST=0
export GCLOCK=875

export GTCI_QUEUE=debug
export GTCI_MPI_NODES=1
export GTCI_MPI_TASKS=4
export GTCI_BUILD_THREADS=12
export GTCI_BUILD_COMMAND="srun -p pp-short -c 12 --time=00:30:00"

export GTCMAKE_CXX_COMPILER=$(which g++)
export GTCMAKE_C_COMPILER=$(which gcc)
export GTCMAKE_FORTRAN_COMPILER=$(which gfortran)
