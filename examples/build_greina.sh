#!/bin/bash

module load cmake/2.8.12
module load /users/crosetto/local/cuda7/7.0.0
module load boost/1.56_gcc4.8.4
module load gcc/4.8.4
module load mpich/ge/gcc/64/3.1
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
export GRIDTOOLS_ROOT_BUILD=$PWD
export GRIDTOOLS_ROOT=$PWD/../

TARGET=$1
REAL_TYPE=$2
if [ "x$TARGET" == "xgpu" ]
then
export USE_GPU=ON
else
export USE_GPU=OFF
fi
echo "USE_GPU=$USE_GPU"

if [ "x$REAL_TYPE" == "xfloat" ]
then
SINGLE_PRECISION=ON
else
SINGLE_PRECISION=OFF
fi
echo "SINGLE_PRECISION=$SINGLE_PRECISION"

RUN_MPI_TESTS=OFF ##$SINGLE_PRECISION

pwd
WHERE_=`pwd`

export JENKINS_COMMUNICATION_TESTS=1

/apps/dom/cmake/repository/gnu_446/bin/cmake \
-DCUDA_NVCC_FLAGS:STRING="-arch=sm_35 -G  -DBOOST_RESULT_OF_USE_TR1_WITH_DECLTYPE_FALLBACK"
-DUSE_GPU:BOOL=$USE_GPU \
-DGTEST_ROOT:PATH=/users/crosetto/gtest-1.7.0 \
-DGNU_COVERAGE:BOOL=OFF \
-DGCL_ONLY:BOOL=OFF \
-DUSE_MPI:BOOL=$RUN_MPI_TESTS \
-DUSE_MPI_COMPILER:BOOL=$RUN_MPI_TESTS  \
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3  -g -fPIC"  \
-DSINGLE_PRECISION:BOOL=$SINGLE_PRECISION \
-DENABLE_CXX11:BOOL=ON \
 ../

make -j8;

if [ "x$TARGET" == "xgpu" ]
then
make tests_gpu;

./build/tests_gpu

  if [ "$RUN_MPI_TESTS" == "ON" ]
  then
      ../examples/communication/run_communication_tests.sh
  fi

else
make tests;
./build/tests

  if [ "$RUN_MPI_TESTS" == "ON" ]
  then
    ../examples/communication/run_communication_tests.sh
  fi

fi
rm -rf *
