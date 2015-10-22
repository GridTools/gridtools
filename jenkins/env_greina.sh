#/bin/sh

module load gcc/4.8.4
#we need a decent cmake version in order to pass the HOST_COMPILER to nvcc
module load /home/cosuna/privatemodules/cmake-3.3.2
module load python/3.4.3
module load boost/1.56_gcc4.8.4
module load mvapich2/gcc/64/2.0-gcc-4.8.2-cuda-6.0
module load cuda70/toolkit/7.0.28
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}

