#!/bin/bash -l

function exit_if_error {
    if [ "x$1" != "x0" ]
    then
        echo "Exit with errors"
        exit $1
    fi
}

module unload CMake
module load daint-gpu
module load cudatoolkit
module rm   PrgEnv-cray
module load CMake

if [[ ${COMPILER} == "gcc" ]]; then
  module load PrgEnv-gnu
  case ${VERSION} in
    "5.3")
      module swap gcc/5.3.0
      ;;
    "6.1")
      module swap gcc/6.1.0
      ;;
    "6.2")
      module swap gcc/6.2.0
      ;;
    "7.1")
      module swap gcc/7.1.0
      ;;
    *)
      module swap gcc/4.9.3
  esac
  export HOST_COMPILER=`which CC`
elif [[ ${COMPILER} == "clang" ]]; then
  module load PrgEnv-gnu
  case ${VERSION} in
    "3.9")
      module load /users/vogtha/modules/compilers/clang/3.9.1
      ;;
    "4.0RC2")
      module load /users/vogtha/modules/compilers/clang/4.0.0rc2
      ;;
    "5.0RC2")
      module load /users/vogtha/modules/compilers/clang/5.0.0rc2
      ;;
    *)
      module load /users/vogtha/modules/compilers/clang/3.8.1
  esac
  export HOST_COMPILER=`which clang++`
elif [[ ${COMPILER} == "icc" ]]; then
  module load PrgEnv-intel
  export HOST_COMPILER=`which icpc`
else
  echo "compiler not supported in environment: ${COMPILER}"
  exit_if_error 444
fi

export BOOST_ROOT=/users/vogtha/boost_1_63_0
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_G2G_PIPELINE=30
export CUDA_ARCH=sm_60
export LAUNCH_MPI_TEST="srun"
export JOB_ENV="export LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST; export MPICH_RDMA_ENABLED_CUDA=1; export MPICH_G2G_PIPELINE=30"
export MPI_HOST_JOB_ENV="export LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST;"
export MPI_CUDA_JOB_ENV="export LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST; export MPICH_RDMA_ENABLED_CUDA=1; export MPICH_G2G_PIPELINE=64"
export MPI_NODES=4
export MPI_TASKS=4
export DEFAULT_QUEUE=normal
export USE_MPI_COMPILER=OFF
export MAKE_THREADS=24
export SRUN_BUILD_COMMAND="srun -C gpu --account c14 --time=00:15:00"
