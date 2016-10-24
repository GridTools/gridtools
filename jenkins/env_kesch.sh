#/bin/bash

if [[ ${VERSION} == "5.3" ]] && [[ "${TARGET}" != "gpu" ]]; then
  module unload GCC/4.9.3-binutils-2.25
  module load mvapich2gdr_gnu/2.1_cuda_7.0
  module load GCC/5.3.0-binutils-2.25
else
  module load PrgEnv-gnu
fi

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
export BOOST_ROOT=/scratch/stefanm/boost_1_62_0/
export BOOST_INCLUDE=/scratch/stefanm/boost_1_62_0/include/
export CUDA_ARCH=sm_37
export DEFAULT_QUEUE=debug
