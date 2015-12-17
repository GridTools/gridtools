#!/bin/bash

function exit_if_error {
    if [ "x$1" != "x0" ]
    then
        echo "Exit with errors"
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
   echo "-f      force build                           "
   echo "-i      build for icosahedral grids           "
   exit 1
}

INITPATH=$PWD
BASEPATH_SCRIPT=$(dirname "${0}")

while getopts "h:b:t:f:c:pzmsi" opt; do
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
    z) FORCE_BUILD="ON"
        ;;
    i) ICOSAHEDRAL_GRID="ON"
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

source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh
if [ $FORCE_BUILD == "ON" ]; then
    rm -rf build
fi
mkdir -p build;
cd build;

#
# full path to the virtual environment where the Python tests run
#
VENV_PATH=${HOME}/venv_gridtools4py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:${VENV_PATH}/lib/python3.4/site-packages/PySide-1.2.2-py3.4-linux-x86_64.egg/PySide

if [ "x$TARGET" == "xgpu" ]; then
    USE_GPU=ON
else
    USE_GPU=OFF
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

if [[ -z ${ICOSAHEDRAL_GRID} ]]; then
    STRUCTURED_GRIDS="ON"
else
    STRUCTURED_GRIDS="OFF"
fi

cmake \
-DBoost_NO_BOOST_CMAKE="true" \
-DCUDA_NVCC_FLAGS:STRING="--relaxed-constexpr" \
-DCUDA_ARCH:STRING="sm_35" \
-DCMAKE_BUILD_TYPE:STRING="$BUILD_TYPE" \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DGPU_ENABLED_FUSION:PATH=../fusion/include \
-DUSE_GPU:BOOL=$USE_GPU \
-DGTEST_LIBRARY:STRING=${GTEST_LIB} \
-DGTEST_MAIN_LIBRARY:STRING=${GTEST_MAINLIB} \
-DGTEST_INCLUDE_DIR:PATH=${GTEST_INC} \
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
-DSTRUCTURED_GRIDS:BOOL=${STRUCTURED_GRIDS} \
 ../

exit_if_error $?

log_file="/tmp/jenkins_${BUILD_TYPE}_${TARGET}_${FLOAT_TYPE}_${CXX_STD}_${PYTHON}_${MPI}.log"
echo "Log file /tmp/jenkins_${BUILD_TYPE}_${TARGET}_${FLOAT_TYPE}_${CXX_STD}_${PYTHON}_${MPI}.log"
if [[ "$SILENT_BUILD" == "ON" ]]; then
    make -j8  >& ${log_file};
    error_code=$?
    if [ ${error_code} -ne 0 ]; then
        cat ${log_file};
        exit_if_error ${error_code}
    fi
else
    make -j8
    exit_if_error $?
fi

bash ${INITPATH}/${BASEPATH_SCRIPT}/test.sh

exit_if_error $?

if [[ "$RUN_MPI_TESTS" == "ON" && ${myhost} == "greina" && ${STRUCTURED_GRIDS} == "ON" ]]
then
   if [ "x$CXX_STD" == "xcxx11" ]
   then
       if [ "x$TARGET" == "xcpu" ]
       then
           mpiexec -np 4 ./build/shallow_water_enhanced 8 8 1 10
           exit_if_error $?

           mpiexec -np 2 ./build/copy_stencil_parallel 62 53 15
           exit_if_error $?
       fi
       if [ "x$TARGET" == "xgpu" ]
       then
            # problems in the execution of the copy_stencil_parallel_cuda
            # TODO fix
            # mpiexec -np 2 ./build/copy_stencil_parallel_cuda 62 53 15
            # exit_if_error $?
            # CUDA allocation error with more than 1 GPU in RELEASE mode
            # To be fixed
            # mpiexec -np 2 ./build/shallow_water_enhanced_cuda 8 8 1 2
            # exit_if_error $?

           mpiexec -np 1 ./build/shallow_water_enhanced_cuda 8 8 1 2
           exit_if_error $?

       fi
       #TODO not updated to greina
       #    ../examples/communication/run_communication_tests.sh
   fi
fi

exit 0
