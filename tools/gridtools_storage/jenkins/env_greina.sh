#/bin/bash

function exit_if_error {
    if [ "x$1" != "x0" ]
    then
        echo "Exit with errors"
        exit $1
    fi
}

if [[ ${COMPILER} == "gcc" ]]; then
  if [[ ${VERSION} == "5.3" ]]; then
      module load GCC/5.3.0
  else
      module load GCC/4.8.4
  fi
elif [[ ${COMPILER} == "clang" ]]; then
  module load Clang/3.7.1-GCC-4.9.3-2.25
else
  echo "compiler not supported in environment: ${COMPILER}"
fi

module load slurm
module load cuda80
module load /users/mbianco/my_modules/cmake-3.5.1
#module load python/3.4.3
#module load mvapich2/gcc/64/2.2-gcc-4.8.4-cuda-7.0
export BOOST_ROOT=/users/stefanm/boost_1_63_0
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export VERBOSE=1
export CUDA_ARCH=sm_35
export DEFAULT_QUEUE=short
