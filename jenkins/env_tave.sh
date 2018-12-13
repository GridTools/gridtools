#!/bin/bash -l

function exit_if_error {
    if [ "x$1" != "x0" ]
    then
        echo "Exit with errors"
        exit $1
    fi
}

module rm   PrgEnv-cray
module rm CMake
module load /users/jenkins/easybuild/tave/modules/all/CMake/3.12.4

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
  export CXX=`which g++`
  export CC=`which gcc`
  export FC=`which gfortran`
elif [[ ${COMPILER} == "icc" ]]; then
  module load PrgEnv-intel
  case ${VERSION} in
    "18")
      module swap intel/18.0.2.199
      ;;
    "17")
      module swap intel/17.0.4.196
      ;;
    *)
      module swap intel/18.0.2.199
  esac
  module load gcc
  export CXX=`which icpc`
  export CC=`which icc`
  export FC=`which ifort`
else
  echo "compiler not supported in environment: ${COMPILER}"
  exit_if_error 444
fi

export BOOST_ROOT=/project/c14/install/kesch/boost/boost_1_67_0 #since it is header only we can use the kesch installation
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export LAUNCH_MPI_TEST="srun"

JOB_ENV=(
    LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST
    )
export HOST_JOB_ENV="${JOB_ENV[*]}"
export CUDA_JOB_ENV="${JOB_ENV[*]}"
export MPI_HOST_JOB_ENV="${JOB_ENV[*]}"
export MPI_CUDA_JOB_ENV="${JOB_ENV[*]}"

export MPI_NODES=4
export MPI_TASKS=4
export DEFAULT_QUEUE=normal
export MAKE_THREADS=16
export GT_ENABLE_BINDINGS_GENERATION=OFF

