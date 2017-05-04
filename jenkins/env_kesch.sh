#/bin/bash

#if [[ -z ${VERSION} ]]; then
#  VERSION="5.3"
#fi
#
#if [[ ${VERSION} == "5.3" ]] && [[ "${TARGET}" != "gpu" ]]; then
#  module unload GCC/4.9.3-binutils-2.25
#  module load GCC/5.3.0-binutils-2.25
#else
#  module load PrgEnv-gnu
module load GCC/4.9.3-binutils-2.25
#fi

module load mvapich2gdr_gnu/2.1_cuda_7.0
module load CMake/3.3.2
module load cudatoolkit

echo $LD_LIBRARY_PATH
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export BOOST_ROOT=/scratch/stefanm/boost_1_62_0/
export BOOST_INCLUDE=/scratch/stefanm/boost_1_62_0/include/
export CUDA_ARCH=sm_37
export DEFAULT_QUEUE=debug
export LAUNCH_MPI_TEST="srun"
export JOB_ENV="export CUDA_AUTO_BOOST=0; export GCLOCK=875; export G2G=1"
export MPI_HOST_JOB_ENV=""
export MPI_CUDA_JOB_ENV="export GCLOCK=875; export CUDA_AUTO_BOOST=0; export G2G=2"
export USE_MPI_COMPILER=ON
export MPI_NODES=1
export MPI_TASKS=4
export CXX=`which g++`
