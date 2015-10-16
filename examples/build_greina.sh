#!/bin/bash

function exit_if_error {
    if [ "x$1" != "x0" ]
    then
        cat /tmp/jenkins_${BUILD_TYPE}_${TARGET}_${FLOAT_TYPE}_${CXX_STD}_${PYTHON}_${MPI}.log;
        echo "Exit with errors"
        rm -rf *
        exit $1
    fi
}

function help {
   echo "$0 [OPTIONS]"
   echo "-h      help"
   echo "-b      build type               [release|debug]"
   echo "-t      target                   [gpu|cpu]"
   echo "-f      floating point precision [float|double]"
   echo "-c      cxx standard             [cxx11|cxx03]"
   echo "-p      activate python                       "
   echo "-m      activate mpi                          "
   echo "-s      activate a silent build               "
   exit 1
}

while getopts "h:b:t:f:c:pms" opt; do
    case "$opt" in
    h|\?)
        help
        exit 0
        ;;
    b) BUILD_TYPE=$OPTARG
        ;;
    t) TARGET=$OPTARG
        ;;
    f) FLOAT_TYPE=$OPTARG
        ;;
    c) CXX_STD=$OPTARG
        ;;
    p) PYTHON="ON"
        ;;
    m) MPI="ON"
        ;;
    s) SILENT_BUILD="ON"
        ;;
    esac
done

if [[ "$BUILD_TYPE" != "debug" ]] && [[ "$BUILD_TYPE" != "release" ]]; then
   help
fi

if [[ "$TARGET" != "gpu" ]] && [[ "$TARGET" != "cpu" ]]; then
   help
fi

if [[ "$FLOAT_TYPE" != "float" ]] && [[ "$FLOAT_TYPE" != "double" ]]; then
   help
fi

if [[ "$CXX_STD" != "cxx11" ]] && [[ "$CXX_STD" != "cxx03" ]]; then
   help
fi


echo $@

if [ -d "build" ]; then
    rm -rf build
fi
mkdir -p build;
cd build;

#
# full path to the virtual environment where the Python tests run
#
VENV_PATH=${HOME}/venv_gridtools4py

#
# environment setup
#
module load gcc/4.8.4
#we need a decent cmake version in order to pass the HOST_COMPILER to nvcc
module load /home/cosuna/privatemodules/cmake-3.3.2
module load python/3.4.3
module load boost/1.56_gcc4.8.4
module load mvapich2/gcc/64/2.0-gcc-4.8.2-cuda-6.0
module load cuda70/toolkit/7.0.28
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:${VENV_PATH}/lib/python3.4/site-packages/PySide-1.2.2-py3.4-linux-x86_64.egg/PySide
export GRIDTOOLS_ROOT_BUILD=$PWD
export GRIDTOOLS_ROOT=$PWD/../
export CUDATOOLKIT_HOME=${CUDA_PATH}

if [ "x$TARGET" == "xgpu" ]; then
  export USE_GPU=ON
else
  export USE_GPU=OFF
fi
echo "USE_GPU=$USE_GPU"

if [[ "$REAL_TYPE" == "float" ]]; then
    SINGLE_PRECISION=ON
else
    SINGLE_PRECISION=OFF
fi
echo "SINGLE_PRECISION=$SINGLE_PRECISION"

if [[ "$CXX_STD" == "cxx11" ]]; then
    CXX_11=ON
else
    CXX_11=OFF
fi
echo "C++ 11 = $CXX_11"

if [[ "$MPI" == "ON" ]]; then
    USE_MPI=ON
else
    USE_MPI=OFF
fi
echo "MPI = $USE_MPI"

if [[ "$PYTHON" == "ON" ]]; then
    USE_PYTHON=ON
else
    USE_PYTHON=OFF
fi
echo "PYTHON = $PYTHON_ON"

RUN_MPI_TESTS=$USE_MPI ##$SINGLE_PRECISION

pwd
WHERE_=`pwd`

export JENKINS_COMMUNICATION_TESTS=1

HOST_COMPILER=`which g++`

cmake \
-DCUDA_NVCC_FLAGS:STRING="--relaxed-constexpr" \
-DCUDA_ARCH:STRING="sm_35" \
-DCMAKE_BUILD_TYPE:STRING="$BUILD_TYPE" \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DGPU_ENABLED_FUSION:PATH=../fusion/include \
-DUSE_GPU:BOOL=$USE_GPU \
-DGTEST_LIBRARY:STRING="/users/crosetto/gtest-1.7.0/libgtest.a" \
-DGTEST_MAIN_LIBRARY:STRING="/users/crosetto/gtest-1.7.0/libgtest_main.a" \
-DGTEST_INCLUDE_DIR:PATH=/users/crosetto/gtest-1.7.0/include \
-DGNU_COVERAGE:BOOL=OFF \
-DGCL_ONLY:BOOL=OFF \
-DCMAKE_CXX_COMPILER="${HOST_COMPILER}" \
-DCMAKE_CXX_FLAGS:STRING="-I${MPI_HOME}/include" \
-DCUDA_HOST_COMPILER:STRING="${HOST_COMPILER}" \
-DUSE_MPI:BOOL=$USE_MPI \
-DUSE_MPI_COMPILER:BOOL=$USE_MPI  \
-DSINGLE_PRECISION:BOOL=$SINGLE_PRECISION \
-DENABLE_CXX11:BOOL=$CXX_11 \
-DENABLE_PYTHON:BOOL=$USE_PYTHON \
-DPYTHON_INSTALL_PREFIX:STRING="${VENV_PATH}" \
-DENABLE_PERFORMANCE_METERS:BOOL=ON \
 ../

exit_if_error $?

make -j8;

exit_if_error $?

echo /tmp/jenkins_${BUILD_TYPE}_${TARGET}_${FLOAT_TYPE}_${CXX_STD}_${PYTHON}_${MPI}.log
if [[ "$SILENT_BUILD" == "ON" ]]; then
    make -j8  >& /tmp/jenkins_${BUILD_TYPE}_${TARGET}_${FLOAT_TYPE}_${CXX_STD}_${PYTHON}_${MPI}.log;
    exit_if_error $?
else
    make -j8
    exit_if_error $?
fi

sh ./run_tests.sh

exit_if_error $?

if [ "$RUN_MPI_TESTS" == "ON" ]
then
    if [ "x$CXX_STD" == "xcxx11" ]
    then
        if [ "x$TARGET" == "xcpu" ]
        then
            mpiexec -np 4 ./build/shallow_water_enhanced 8 8 1 2
            exit_if_error $?

            mpiexec -np 2 ./build/copy_stencil_parallel 62 53 15
            exit_if_error $?
        fi
        if [ "x$TARGET" == "xgpu" ]
        then
            if [[ "$BUILD_TYPE" == "debug" ]] ; then
                mpiexec -np 2 ./build/shallow_water_enhanced_cuda 8 8 1 2
                exit_if_error $?

                # problems in the execution of the copy_stencil_parallel_cuda
                # TODO fix
                # mpiexec -np 2 ./build/copy_stencil_parallel_cuda 62 53 15
                # exit_if_error $?
            else
                # CUDA allocation error with more than 1 GPU in RELEASE mode
                # (works in debug mode). To be fixed
                mpiexec -np 1 ./build/shallow_water_enhanced_cuda 8 8 1 2
                exit_if_error $?

                # problems in the execution of the copy_stencil_parallel_cuda
                # TODO fix
                # mpiexec -np 1 ./build/copy_stencil_parallel_cuda 62 53 15
                # exit_if_error $?
            fi
        fi
        #TODO not updated to greina
        #    ../examples/communication/run_communication_tests.sh
    fi
fi

exit 0
