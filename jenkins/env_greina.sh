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
  exit_if_error 444
fi

module load slurm
module load cuda80/8.0.27
module load /users/mbianco/my_modules/cmake-3.5.1
module load /users/mbianco/my_modules/boost-1.59
#module load python/3.4.3
#module load mvapich2/gcc/64/2.2-gcc-4.8.4-cuda-7.0
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}

if [[ ${VERSION} == "5.3" ]] && [[ ${COMPILER} == "gcc" ]]; then
    export GTEST_LIB=/users/cosuna/software/googletest/install/5.3/lib/libgtest.a
    export GTEST_MAINLIB=/users/cosuna/software/googletest/install/5.3/lib/libgtest_main.a
    export GTEST_INC=/users/cosuna/software/googletest/install/5.3/include 
else
    export GTEST_LIB=/users/crosetto/gtest-1.7.0/libgtest.a
    export GTEST_MAINLIB=/users/crosetto/gtest-1.7.0/libgtest_main.a
    export GTEST_INC=/users/crosetto/gtest-1.7.0/include

fi

export CUDA_ARCH=sm_35
export CUDA_VERSION=75
export DEFAULT_QUEUE=k80
