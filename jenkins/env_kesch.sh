#/bin/bash

module load craype-network-infiniband
module load craype-haswell
module load craype-accel-nvidia35
module load cray-libsci
module load cudatoolkit/8.0.61
module load mvapich2gdr_gnu/2.2_cuda_8.0
module load gcc/5.4.0-2.26
module load cmake/3.9.1

export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export BOOST_ROOT=/users/vogtha/boost_1_65_1
export BOOST_INCLUDE=/users/vogtha/boost_1_65_1/include/
export CUDA_ARCH=sm_37
export DEFAULT_QUEUE=debug
export LAUNCH_MPI_TEST="srun"
export JOB_ENV="export CUDA_AUTO_BOOST=0; export GCLOCK=875; export G2G=1; export LD_PRELOAD=/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5/lib64/libmpi.so"
export MPI_HOST_JOB_ENV=""
export MPI_CUDA_JOB_ENV="export GCLOCK=875; export CUDA_AUTO_BOOST=0; export G2G=2; export LD_PRELOAD=/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5/lib64/libmpi.so"
export USE_MPI_COMPILER=ON
export MPI_NODES=1
export MPI_TASKS=4
export CXX=`which g++`
