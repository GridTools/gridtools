#/bin/bash

function exit_if_error {
    if [ "x$1" != "x0" ]
    then
        echo "Exit with errors"
        exit $1
    fi
}

module load CMake
module load cudatoolkit/8.0.34_2.2.5_g8ce7a9a-2.1
export BOOST_ROOT=/project/csstaff/mbianco/boost_1_62_0
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
  echo "compiler not supported in environment: ${COMPILER}"
  exit_if_error 444
else
  echo "compiler not supported in environment: ${COMPILER}"
  exit_if_error 444
fi


#module load python/3.4.3
#module load mvapich2/gcc/64/2.2-gcc-4.8.4-cuda-7.0
#export Boost_NO_SYSTEM_PATHS=true
#export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_G2G_PIPELINE=30
export CUDA_ARCH=sm_35
export LAUNCH_MPI_TEST="srun"
export JOB_ENV="export LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST; module swap gcc/5.3.0; export MPICH_RDMA_ENABLED_CUDA=1; export MPICH_G2G_PIPELINE=30"
export MPI_HOST_JOB_ENV="export LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST;"
export MPI_CUDA_JOB_ENV="export LAUNCH_MPI_TEST=$LAUNCH_MPI_TEST; export MPICH_RDMA_ENABLED_CUDA=1; export MPICH_G2G_PIPELINE=64"
export MPI_NODES=4
export MPI_TASKS=4
export DEFAULT_QUEUE=normal
export USE_MPI_COMPILER=OFF
export HOST_COMPILER=CC
