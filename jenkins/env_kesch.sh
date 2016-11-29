#/bin/bash

module load craype-haswell
module load craype-network-infiniband
module load mvapich2gdr_gnu/2.1_cuda_7.0
module load GCC/4.9.3-binutils-2.25
module load cudatoolkit/7.0.28
#we need a decent cmake version in order to pass the HOST_COMPILER to nvcc
module load CMake/3.3.2


echo $LD_LIBRARY_PATH
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export GTEST_LIB=/scratch/cosuna/software/gtest-1.7.0/lib/libgtest.a
export GTEST_MAINLIB=/scratch/cosuna/software/gtest-1.7.0/lib/libgtest_main.a
export GTEST_INC=/scratch/cosuna/software/gtest-1.7.0/include
export BOOST_ROOT=/scratch/cosuna/software/boost_1_59_0/
export BOOST_INCLUDE=/scratch/cosuna/software/boost_1_59_0/include/
export CUDA_ARCH=sm_37
export DEFAULT_QUEUE=debug
export LAUNCH_MPI_TEST="srun"
export JOB_ENV="export ENABLE_CUDA=1; export CUDA_AUTO_BOOST=0; export GCLOCK=875; export G2G=1; export CUDA_AUTO_BOOST=0;"
export USE_MPI_COMPILER=ON
export MPI_NODES=1
export MPI_TASKS=4
export CXX=`which g++`
