#!/bin/bash -l

function exit_if_error {
    if [ "x$1" != "x0" ]
    then
        echo "Exit with errors"
        exit $1
    fi
}

module load daint-gpu
module load cudatoolkit/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52
module rm   PrgEnv-cray
module rm CMake
module load /users/jenkins/easybuild/daint/haswell/modules/all/CMake/3.12.4


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
    "7.3")
      module swap gcc/7.3.0
      ;;
    *)
      module swap gcc/7.3.0
  esac
  export HOST_COMPILER=`which CC`
elif [[ ${COMPILER} == "clang" ]]; then
  case ${VERSION} in
    "5.0RC2")
      module load /users/vogtha/modules/compilers/clang/5.0.0rc2
      ;;
    "7.0")
      module load /users/vogtha/modules/compilers/clang/7.0.1
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

export CPATH=$CPATH:$MPICH_DIR/include
export BOOST_ROOT=$SCRATCH/../jenkins/install/boost/boost_1_67_0
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_G2G_PIPELINE=30
export CUDA_ARCH=sm_60
export LAUNCH_MPI_TEST="srun"

JOB_ENV_ARR=(
    LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST
    MPICH_RDMA_ENABLED_CUDA=1
    MPICH_G2G_PIPELINE=30
    )
MPI_HOST_JOB_ENV_ARR=(
    LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST
    )
MPI_CUDA_JOB_ENV_ARR=(
    LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST
    MPICH_RDMA_ENABLED_CUDA=1
    MPICH_G2G_PIPELINE=64)
export HOST_JOB_ENV="${JOB_ENV_ARR[*]}"
export CUDA_JOB_ENV="${JOB_ENV_ARR[*]}"
export MPI_HOST_JOB_ENV="${MPI_HOST_JOB_ENV_ARR[*]}"
export MPI_CUDA_JOB_ENV="${MPI_CUDA_JOB_ENV_ARR[*]}"

export MPI_NODES=4
export MPI_TASKS=4
export DEFAULT_QUEUE=normal
export MAKE_THREADS=24
export SRUN_BUILD_COMMAND="srun -C gpu --account c14 -p cscsci --time=00:20:00"

