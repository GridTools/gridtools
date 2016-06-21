#/bin/sh

module load PrgEnv-gnu
#we need a decent cmake version in order to pass the HOST_COMPILER to nvcc
module load cmake
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
