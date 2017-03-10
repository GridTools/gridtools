#/bin/bash

function exit_if_error {
    if [ "x$1" != "x0" ]
    then
        echo "Exit with errors"
        exit $1
    fi
}

module unload CMake
module load daint-gpu
module load /users/vogtha/modules/CMake/3.7.2
module load cudatoolkit
module rm   PrgEnv-cray
module load PrgEnv-gnu/6.0.3
export BOOST_ROOT=/apps/daint/UES/jenkins/dom-acceptance/haswell/easybuild/software/Boost/1.61.0-CrayGNU-2016.11-Python-2.7.12/

if [[ ${COMPILER} == "gcc" ]]; then
  if [[ ${VERSION} == "5.3" ]]; then
      module swap gcc/5.3.0
  else
    if [[ ${VERSION} == "6.1" ]]; then
        module swap gcc/6.1.0
    else
        module swap gcc/4.9.3
    fi
  fi
elif [[ ${COMPILER} == "clang" ]]; then
  module unload PrgEnv-gnu
  module load /users/vogtha/modules/compilers/clang/3.8.1
else
  echo "compiler not supported in environment: ${COMPILER}"
  exit_if_error 444
fi


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
export HOST_COMPILER=CC
