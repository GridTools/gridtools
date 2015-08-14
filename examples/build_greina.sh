#!/bin/bash


#
# full path to the virtual environment where the Python tests run
#
VENV_PATH=${HOME}/venv_gridtools4py

#
# environment setup
#
module load gcc/4.8.4
module load cmake/2.8.12
module load python/3.4.3
module load boost/1.56_gcc4.8.4
module load mpich/ge/gcc/64/3.1
module load /users/crosetto/local/cuda7/7.0.0
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:${VENV_PATH}/lib/python3.4/site-packages/PySide-1.2.2-py3.4-linux-x86_64.egg/PySide
export GRIDTOOLS_ROOT_BUILD=$PWD
export GRIDTOOLS_ROOT=$PWD/../
export CUDATOOLKIT_HOME=${CUDA_ROOT}

TARGET=$1
REAL_TYPE=$2
CXX_11_ON=$3
MPI=$4
PYTHON_ON=$5

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

if [ "x$CXX_11_ON" == "xcxx11" ]
then
CXX_11=ON
else
CXX_11=OFF
fi
echo "C++ 11 = $CXX_11"

if [ "x$MPI" == "xMPI" ]
then
USE_MPI=ON
else
USE_MPI=OFF
fi
echo "MPI = $USE_MPI"

if [ "x$PYTHON_ON" == "xpython_on" ]
then
USE_PYTHON=ON
else
USE_PYTHON=OFF
fi
echo "PYTHON = $PYTHON_ON"

RUN_MPI_TESTS=ON ##$SINGLE_PRECISION

pwd
WHERE_=`pwd`

export JENKINS_COMMUNICATION_TESTS=1

cmake \
-DCUDA_ARCH:STRING="sm_35" \
-DCMAKE_BUILD_TYPE:STRING="DEBUG" \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DUSE_GPU:BOOL=$USE_GPU \
-DGTEST_LIBRARY:STRING="/users/crosetto/gtest-1.7.0/libgtest.a" \
-DGTEST_MAIN_LIBRARY:STRING="/users/crosetto/gtest-1.7.0/libgtest_main.a" \
-DGTEST_INCLUDE_DIR:PATH=/users/crosetto/gtest-1.7.0/include \
-DGNU_COVERAGE:BOOL=OFF \
-DGCL_ONLY:BOOL=OFF \
-DCMAKE_CXX_COMPILER="/cm/shared/apps/mpich/ge/gcc/64/3.1/bin/mpicxx" \
-DCMAKE_C_COMPILER="/cm/shared/apps/mpich/ge/gcc/64/3.1/bin/mpicc" \
-DUSE_MPI:BOOL=$USE_MPI \
-DUSE_MPI_COMPILER:BOOL=OFF  \
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3  -g -fPIC -DBOOST_RESULT_OF_USE_TR1"  \
-DSINGLE_PRECISION:BOOL=$SINGLE_PRECISION \
-DENABLE_CXX11:BOOL=$CXX_11 \
-DENABLE_PYTHON:BOOL=$USE_PYTHON \
-DPYTHON_INSTALL_PREFIX:STRING="${VENV_PATH}" \
 ../

make -j8;

sh ./run_tests.sh

if [ "x$TARGET" == "xcpu" ]
then
    if [ "$RUN_MPI_TESTS" == "ON" ]
    then
        if [ "x$CXX_11_ON" == "xcxx11" ]
        then
            mpiexec -np 4 ./build/shallow_water_enhanced 8 8 1 2
        fi

        #TODO not updated to greina
        #    ../examples/communication/run_communication_tests.sh
    fi
fi
rm -rf *
