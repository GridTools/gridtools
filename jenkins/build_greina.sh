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
   echo "-l      compiler                 [gcc|clang]  "
   echo "-m      activate mpi                          "
   echo "-s      activate a silent build               "
   echo "-z      force build                           "
   echo "-d      do not clean build                    "
   echo "-q      queue for testing                     "
   echo "-x      compiler version                      "
   exit 1
}

INITPATH=$PWD
BASEPATH_SCRIPT=$(dirname "${0}")
FORCE_BUILD=OFF
VERBOSE_RUN="OFF"
VERSION="4.9"

while getopts "h:b:t:f:c:l:pzmsidvq:x:" opt; do
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
    d) DONOTCLEAN="ON"
        ;;
    l) export COMPILER=$OPTARG
        ;;
    v) VERBOSE_RUN="ON"
        ;;
    q) QUEUE=$OPTARG
        ;;
    x) VERSION=$OPTARG
        ;;
    esac
done

if [[ "$VERSION"  != "4.9" ]] && [[ "$VERSION" != "5.3" ]]; then
    echo "VERSION $VERSION not supported"
    help
fi

if [[ "$BUILD_TYPE" != "debug" ]] && [[ "$BUILD_TYPE" != "release" ]]; then
   help
fi

if [[ "$TARGET" != "gpu" ]] && [[ "$TARGET" != "cpu" ]]; then
   help
fi

if [[ "$TARGET"  == "gpu" ]] && [[ "$VERSION" != "4.9" ]]; then
    echo "VERSION $VERSION not supported for gpu"
    help
fi

echo $@

source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh
if [ "x$FORCE_BUILD" == "xON" ]; then
    echo Deleting all
    test -e build
    if [ $? -ne 0 ] ; then
        rm -rf build
    fi
fi

mkdir -p build;
cd build;

if [ "x$TARGET" == "xgpu" ]; then
    USE_GPU=ON
else
    USE_GPU=OFF
fi
echo "USE_GPU=$USE_GPU"

if [[ "$MPI" == "ON" ]]; then
    USE_MPI=ON
else
    USE_MPI=OFF
fi
echo "MPI = $USE_MPI"

if [[ "$BUILD_TYPE" == "debug" ]]; then
    DEBUG=1
    ADDITIONAL_FLAGS="-DDEBUG $ADDITIONAL_FLAGS"
else
    DEBUG=0
fi
echo "DEBUG = $DEBUG"

RUN_MPI_TESTS=$USE_MPI ##$SINGLE_PRECISION

pwd
WHERE_=`pwd`

export JENKINS_COMMUNICATION_TESTS=1

if [[ ${COMPILER} == "gcc" ]] ; then
    HOST_COMPILER=`which g++`
elif [[ ${COMPILER} == "clang" ]] ; then
    HOST_COMPILER=`which clang++`
    ADDITIONAL_FLAGS="-ftemplate-depth=1024 $ADDITIONAL_FLAGS"
    if [[ ${USE_GPU} == "ON" ]]; then
       echo "Clang not supported with nvcc"
       exit_if_error 334
    fi
else
    echo "COMPILER ${COMPILER} not supported"
    exit_if_error 333
fi


# echo "Printing ENV"
# env
cmake \
-DBoost_NO_BOOST_CMAKE="true" \
-DBOOST_ROOT=$BOOST_ROOT \
-DBOOST_LIBRARYDIR=$BOOST_ROOT/lib \
-DBOOST_INCLUDEDIR=$BOOST_ROOT/include \
-DCUDA_ARCH:STRING="$CUDA_ARCH" \
-DCMAKE_BUILD_TYPE:STRING="$BUILD_TYPE" \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DGPU_ENABLED_FUSION:PATH=../fusion/include \
-DUSE_GPU:BOOL=$USE_GPU \
-DDEBUG:BOOL=$DEBUG \
-DGCL_ONLY:BOOL=OFF \
-DCMAKE_CXX_COMPILER="${HOST_COMPILER}" \
-DCMAKE_CXX_FLAGS:STRING="-I${MPI_HOME}/include ${ADDITIONAL_FLAGS}" \
-DCUDA_HOST_COMPILER:STRING="${HOST_COMPILER}" \
-DUSE_MPI:BOOL=$USE_MPI \
-DUSE_MPI_COMPILER:BOOL=$USE_MPI  \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 ../

exit_if_error $?

#number of trials for compilation. We add this here because sometime intermediate links of nvcc are missing
#some object files, probably related to parallel make compilation, but we dont know yet how to solve this.
#Workaround here is to try multiple times the compilation step
num_make_rep=2

error_code=0
log_file="/tmp/jenkins_${BUILD_TYPE}_${TARGET}_${FLOAT_TYPE}_${CXX_STD}_${PYTHON}_${MPI}_${RANDOM}.log"
if [[ "$SILENT_BUILD" == "ON" ]]; then
    echo "Log file ${log_file}"
    for i in `seq 1 $num_make_rep`;
    do
      echo "COMPILATION # ${i}"
      if [ ${i} -eq ${num_make_rep} ]; then
          make  >& ${log_file};
      else
          make -j5  >& ${log_file};
      fi
      error_code=$?
      if [ ${error_code} -eq 0 ]; then
          break # Skip the make repetitions
      fi
    done

    if [ ${error_code} -ne 0 ]; then
        cat ${log_file};
    fi
else
    make -j10
    error_code=$?
fi

if [[ -z ${DONOTCLEAN} ]]; then
    test -e ${log_file}
    if [ $? -eq 0 ] ; then
       rm ${log_file}
    fi
fi

exit_if_error ${error_code}
module load slurm
if [[ ${myhost} == "kesch" ]]; then
    srun --partition=debug --gres=gpu:1 make test && hostname
elif [[ ${myhost} == "daint" ]]; then
    srun --partition=normal --account=c01 -C gpu make test && hostname
else
    srun --partition=short --gres=gpu:1 make test && hostname
fi
exit_if_error $?

