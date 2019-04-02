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
   echo "-b      build type               [release|debug] "
   echo "-t      target                   [gpu|cpu]       "
   echo "-f      floating point precision [float|double]  "
   echo "-l      compiler                 [gcc|clang|icc] "
   echo "-m      activate mpi                             "
   echo "-s      activate a silent build                  "
   echo "-z      force build                              "
   echo "-i      build for icosahedral grids              "
   echo "-d      do not clean build                       "
   echo "-q      queue for testing                        "
   echo "-x      compiler version                         "
   echo "-n      execute the build on a compute node      "
   echo "-k      build only the given Makefile targets    "
   echo "-o      compile only (not tests are run)         "
   echo "-p      enable performance testing               "
   echo "-C      Only run CMAKE configure and generation  "
   exit 1
}

INITPATH=$PWD
BASEPATH_SCRIPT=$(dirname "${0}")
ABSOLUTEPATH_SCRIPT=${INITPATH}/${BASEPATH_SCRIPT#$INITPATH}
FORCE_BUILD=OFF
VERSION_="5.3"
GENERATE_ONLY="OFF"
PERFORMANCE_TESTING="OFF"

while getopts "hb:t:f:l:zmsidvq:x:incok:pCI:" opt; do
    case "$opt" in
    h|\?)
        help
        exit 0
        ;;
    b) BUILD_TYPE=$OPTARG
        ;;
    t) TARGET_=$OPTARG
        ;;
    f) FLOAT_TYPE=$OPTARG
        ;;
    m) MPI="ON"
        ;;
    s) SILENT_BUILD="ON"
        ;;
    z) FORCE_BUILD="ON"
        ;;
    i) ICOSAHEDRAL_GRID="ON"
        ;;
    d) DONOTCLEAN="ON"
        ;;
    l) export COMPILER=$OPTARG
        ;;
    q) QUEUE=$OPTARG
        ;;
    x) VERSION_=$OPTARG
        ;;
    n) BUILD_ON_CN="ON"
        ;;
    k) MAKE_TARGETS="$MAKE_TARGETS $OPTARG"
        ;;
    o) COMPILE_ONLY="ON"
        ;;
    p) PERFORMANCE_TESTING="ON"
        ;;
    C) GENERATE_ONLY="ON"
        ;;
    I) GRIDTOOLS_INSTALL_PATH=$OPTARG
        ;;
    esac
done

export VERSION=${VERSION_}

if [[ "$BUILD_TYPE" != "debug" ]] && [[ "$BUILD_TYPE" != "release" ]]; then
   help
fi

if [[ "$TARGET_" != "gpu" ]] && [[ "$TARGET_" != "cpu" ]]; then
   help
fi
export TARGET=${TARGET_}

if [[ "$FLOAT_TYPE" != "float" ]] && [[ "$FLOAT_TYPE" != "double" ]]; then
   help
fi

echo $@

source ${ABSOLUTEPATH_SCRIPT}/machine_env.sh
source ${ABSOLUTEPATH_SCRIPT}/env_${myhost}.sh

if [[ -z ${GRIDTOOLS_INSTALL_PATH} ]]; then
    GRIDTOOLS_INSTALL_PATH=$INITPATH/install
fi
if [[ -z ${MAKE_TARGETS} ]]; then
    MAKE_TARGETS=install
fi

echo "BOOST_ROOT=$BOOST_ROOT"

if [ "x$FORCE_BUILD" == "xON" ]; then
    echo Deleting all
    test -e build
    if [ $? -eq 0 ] ; then
        echo "REMOVING ALL FILES"
        rm -rf build
    fi
fi

mkdir -p build;
cd build;

if [ "x$TARGET" == "xgpu" ]; then
    ENABLE_X86=OFF
    ENABLE_NAIVE=OFF
    ENABLE_CUDA=ON
    ENABLE_MC=OFF
else
    ENABLE_X86=ON
    ENABLE_NAIVE=ON
    ENABLE_CUDA=OFF
    if [[ -z ${ICOSAHEDRAL_GRID} ]]; then
        ENABLE_MC=ON
    else
        ENABLE_MC=OFF
    fi
fi
echo "ENABLE_CUDA=$ENABLE_CUDA"
echo "ENABLE_X86=$ENABLE_X86"
echo "ENABLE_NAIVE=$ENABLE_NAIVE"
echo "ENABLE_MC=$ENABLE_MC"

if [[ "$FLOAT_TYPE" == "float" ]]; then
    GT_SINGLE_PRECISION=ON
else
    GT_SINGLE_PRECISION=OFF
fi
echo "GT_SINGLE_PRECISION=$GT_SINGLE_PRECISION"

if [[ "$MPI" == "ON" ]]; then
    USE_MPI=ON
else
    USE_MPI=OFF
fi
echo "MPI = $USE_MPI"

RUN_MPI_TESTS=$USE_MPI ##$SINGLE_PRECISION

pwd
WHERE_=`pwd`

export JENKINS_COMMUNICATION_TESTS=1

if [[ -z ${ICOSAHEDRAL_GRID} ]]; then
    GT_ICOSAHEDRAL_GRIDS="OFF"
else
    GT_ICOSAHEDRAL_GRIDS="ON"
fi

# measuring time
export START_TIME=$SECONDS

if [[ "$BUILD_ON_CN" == "ON" ]]; then
    if [[ -z ${SRUN_BUILD_COMMAND} ]]; then
        echo "No command for building on a compute node available, falling back to normal mode."
        SRUN_BUILD_COMMAND=""
    else
        echo "Building on a compute node (launching from `hostname`)"
    fi
else
    echo "Building on `hostname`"
    SRUN_BUILD_COMMAND=""
fi

if [[ -z ${GT_ENABLE_BINDINGS_GENERATION} ]]; then
	GT_ENABLE_BINDINGS_GENERATION=ON
fi

cmake \
-DCMAKE_INSTALL_PREFIX="${GRIDTOOLS_INSTALL_PATH}" \
-DBoost_NO_BOOST_CMAKE="true" \
-DCMAKE_BUILD_TYPE:STRING="$BUILD_TYPE" \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DGT_ENABLE_BACKEND_X86:BOOL=$ENABLE_X86 \
-DGT_ENABLE_BACKEND_NAIVE:BOOL=$ENABLE_NAIVE \
-DGT_ENABLE_BACKEND_CUDA:BOOL=$ENABLE_CUDA \
-DGT_ENABLE_BACKEND_MC:BOOL=$ENABLE_MC \
-DGT_GCL_ONLY:BOOL=OFF \
-DCMAKE_CXX_COMPILER="${HOST_COMPILER}" \
-DCMAKE_CXX_FLAGS:STRING="-I${MPI_HOME}/include ${ADDITIONAL_FLAGS}" \
-DCMAKE_CUDA_HOST_COMPILER:STRING="${HOST_COMPILER}" \
-DGT_USE_MPI:BOOL=$USE_MPI \
-DGT_SINGLE_PRECISION:BOOL=$GT_SINGLE_PRECISION \
-DGT_ENABLE_PERFORMANCE_METERS:BOOL=ON \
-DGT_TESTS_ICOSAHEDRAL_GRID:BOOL=${GT_ICOSAHEDRAL_GRIDS} \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DBOOST_ROOT=$BOOST_ROOT \
-DGT_ENABLE_BINDINGS_GENERATION=$GT_ENABLE_BINDINGS_GENERATION \
-DGT_ENABLE_PYUTILS=$PERFORMANCE_TESTING \
-DGT_TESTS_REQUIRE_FORTRAN_COMPILER=ON \
-DGT_TESTS_REQUIRE_C_COMPILER=ON \
-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON \
../

echo "
cmake \
-DBoost_NO_BOOST_CMAKE=\"true\" \
-DCMAKE_BUILD_TYPE:STRING=\"$BUILD_TYPE\" \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DGT_ENABLE_BACKEND_X86:BOOL=$ENABLE_X86 \
-DGT_ENABLE_BACKEND_NAIVE:BOOL=$ENABLE_NAIVE \
-DGT_ENABLE_BACKEND_CUDA:BOOL=$ENABLE_CUDA \
-DGT_ENABLE_BACKEND_MC:BOOL=$ENABLE_MC \
-DGT_GCL_ONLY:BOOL=OFF \
-DCMAKE_CXX_COMPILER=\"${HOST_COMPILER}\" \
-DCMAKE_CXX_FLAGS:STRING=\"-I${MPI_HOME}/include ${ADDITIONAL_FLAGS}\" \
-DCMAKE_CUDA_HOST_COMPILER:STRING=\"${HOST_COMPILER}\" \
-DGT_USE_MPI:BOOL=$USE_MPI \
-DGT_SINGLE_PRECISION:BOOL=$GT_SINGLE_PRECISION \
-DGT_ENABLE_PERFORMANCE_METERS:BOOL=ON \
-DGT_TESTS_ICOSAHEDRAL_GRID:BOOL=${GT_ICOSAHEDRAL_GRIDS} \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DBOOST_ROOT=$BOOST_ROOT \
-DGT_ENABLE_PYUTILS=$PERFORMANCE_TESTING \
-DGT_TESTS_REQUIRE_FORTRAN_COMPILER=ON \
-DGT_TESTS_REQUIRE_C_COMPILER=ON \
-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON \
../
"

if [ "x$GENERATE_ONLY" == "xON" ]; then
    exit 0
fi

exit_if_error $?

#number of trials for compilation. We add this here because sometime intermediate links of nvcc are missing
#some object files, probably related to parallel make compilation, but we dont know yet how to solve this.
#Workaround here is to try multiple times the compilation step
num_make_rep=2
error_code=0
log_file="/tmp/jenkins_${BUILD_TYPE}_${TARGET}_${FLOAT_TYPE}_${CXX_STD}_${MPI}_${RANDOM}.log"

if [[ -z ${MAKE_THREADS} ]]; then
    MAKE_THREADS=5
fi

if [[ "$SILENT_BUILD" == "ON" ]]; then
    echo "Log file ${log_file}"
    for i in `seq 1 $num_make_rep`;
    do
      echo "COMPILATION # ${i}"
      ${SRUN_BUILD_COMMAND} nice make -j${MAKE_THREADS} ${MAKE_TARGETS} >& ${log_file};

      error_code=$?
      if [ ${error_code} -eq 0 ]; then
          break # Skip the make repetitions
      fi
    done

    nwarnings=`grep -i "warning" ${log_file} | wc -l`
    if [[ ${TARGET} != "cpu" ]]; then
        if [ ${nwarnings} -ne 0 ]; then
            echo "Treating warnings as errors! Build failed because of ${nwarnings} warnings!"
            error_code=$((error_code || `echo "1"` ))
        fi
    fi

    if [ ${error_code} -ne 0 ]; then
        cat ${log_file};
    fi
else
    ${SRUN_BUILD_COMMAND} nice make -j${MAKE_THREADS} ${MAKE_TARGETS}
    error_code=$?
fi

if [[ -z ${DONOTCLEAN} ]]; then
    test -e ${log_file}
    if [ $? -eq 0 ] ; then
       rm ${log_file}
    fi
fi

exit_if_error ${error_code}

if  [[ "$COMPILE_ONLY" == "ON" ]]; then
    exit 0;
fi

queue_str=""
if [[ ${QUEUE} ]] ; then
  queue_str="-q ${QUEUE}"
fi


if [[ "$RUN_MPI_TESTS" == "ON" ]]; then
    bash ${ABSOLUTEPATH_SCRIPT}/test.sh ${queue_str} -m $RUN_MPI_TESTS -n $MPI_NODES -t $MPI_TASKS -g $ENABLE_CUDA
else
    bash ${ABSOLUTEPATH_SCRIPT}/test.sh ${queue_str}
fi

exit_if_error $?

# test installation by building and executing examples
if [[ "$MAKE_TARGETS" == "install" ]]; then # only if GT was installed
    mkdir -p build_examples && cd build_examples
    cmake ${GRIDTOOLS_INSTALL_PATH}/gridtools_examples \
        -DCMAKE_BUILD_TYPE:STRING=\"$BUILD_TYPE\" \
        -DCMAKE_CXX_COMPILER="${HOST_COMPILER}" \
        -DCMAKE_CUDA_HOST_COMPILER:STRING="${HOST_COMPILER}" \
        -DGT_EXAMPLES_FORCE_CUDA:BOOL=$ENABLE_CUDA

    if [[ "$SILENT_BUILD" == "ON" ]]; then
        echo "Log file ${log_file}"

        ${SRUN_BUILD_COMMAND} nice make -j${MAKE_THREADS} >& ${log_file};
        error_code=$?
        if [ ${error_code} -ne 0 ]; then
            cat ${log_file};
        fi
    else
        ${SRUN_BUILD_COMMAND} nice make -j${MAKE_THREADS}
        error_code=$?
    fi
    exit_if_error $error_code

    bash ${ABSOLUTEPATH_SCRIPT}/test.sh ${queue_str} -s "make test"
    exit_if_error $?
fi

exit 0
