#/bin/bash

module load PrgEnv-gnu
#we need a decent cmake version in order to pass the HOST_COMPILER to nvcc
module load CMake/3.3.2
#module load python/3.4.3
#module load boost/1.56_gcc4.8.4
#module load mvapich2/gcc/64/2.0-gcc-4.8.2-cuda-6.0
module load cudatoolkit
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
export JOB_ENV="export CUDA_AUTO_BOOST=0; export GCLOCK=875; export G2G=1; export CUDA_AUTO_BOOST=0;"
export DEFAULT_QUEUE=dev
export USE_MPI_COMPILER=ON
export MPI_NODES=1
export MPI_TASKS=4
