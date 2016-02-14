#/bin/sh

#we need a decent cmake version in order to pass the HOST_COMPILER to nvcc
module load /home/cosuna/privatemodules/cmake-3.3.2
#module load cmake
module load python/3.4.3
module load boost/1.56_gcc4.8.4
module load mvapich2/gcc/64/2.2-gcc-4.8.4-cuda-7.0
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export GTEST_LIB=/users/crosetto/gtest-1.7.0/libgtest.a
export GTEST_MAINLIB=/users/crosetto/gtest-1.7.0/libgtest_main.a
export GTEST_INC=/users/crosetto/gtest-1.7.0/include
export CUDA_ARCH=sm_35
