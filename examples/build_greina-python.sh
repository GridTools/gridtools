#!/bin/bash


#
# environment setup
#
module load gcc/4.8.4
module load cmake/2.8.9
module load python/3.4.3
module load boost/1.56_gcc4.8.4
module load cuda70/toolkit/7.0.28

export CXX=$( which g++ )
export GRIDTOOLS_ROOT=$PWD/../
export CUDATOOLKIT_HOME=/cm/shared/apps/cuda70/toolkit/7.0.28

USE_GPU=ON
echo "USE_GPU=$USE_GPU"

SINGLE_PRECISION=OFF
echo "SINGLE_PRECISION=$SINGLE_PRECISION"

CXX_11=OFF
echo "C++ 11 = $CXX_11"

pwd
WHERE_=`pwd`

cmake \
-DCUDA_NVCC_FLAGS:STRING="-arch=sm_35 -O3 -DNDEBUG " \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DUSE_GPU:BOOL=$USE_GPU \
-DGTEST_LIBRARY:STRING="/users/crosetto/gtest-1.7.0/libgtest.a" \
-DGTEST_MAIN_LIBRARY:STRING="/users/crosetto/gtest-1.7.0/libgtest_main.a" \
-DGTEST_INCLUDE_DIR:PATH=/users/crosetto/gtest-1.7.0/include \
-DGNU_COVERAGE:BOOL=OFF \
-DGCL_ONLY:BOOL=OFF \
-DUSE_MPI:BOOL=OFF \
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3  -g -fPIC -DBOOST_RESULT_OF_USE_TR1"  \
-DSINGLE_PRECISION:BOOL=$SINGLE_PRECISION \
-DENABLE_CXX11:BOOL=$CXX_11 \
-DENABLE_PYTHON:BOOL=ON \
-DPYTHON_INSTALL_PREFIX:STRING="${WHERE_}/gridtools4py" \
 ../

make python_tests

#if [ "x$TARGET" == "xgpu" ]
#then
#make tests_gpu;
#
#./build/tests_gpu
#
##  if [ "$RUN_MPI_TESTS" == "ON" ]
##  then
#      #TODO not updated to greina
#      # ../examples/communication/run_communication_tests.sh
##  fi
#
#else
#make tests;
#./build/tests
#
#fi
#rm -rf *
