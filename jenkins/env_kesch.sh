#/bin/bash

module load PrgEnv-gnu
#we need a decent cmake version in order to pass the HOST_COMPILER to nvcc
module load CMake/3.3.2 
module load cudatoolkit
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export BOOST_ROOT=/scratch/stefanm/boost_1_62_0/
export CUDA_ARCH=sm_37
export DEFAULT_QUEUE=debug
