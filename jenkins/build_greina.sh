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
   echo "-l      compiler                 [gcc|clang]  "
   echo "-m      activate mpi                          "
   echo "-s      activate a silent build               "
   echo "-z      force build                           "
   echo "-i      build for icosahedral grids           "
   echo "-d      do not clean build                    "
   echo "-v      compile in VERBOSE mode               "
   echo "-q      queue for testing                     "
   echo "-x      compiler version                      "
   exit 1
}

INITPATH=$PWD
BASEPATH_SCRIPT=$(dirname "${0}")
FORCE_BUILD=OFF
VERBOSE_RUN="OFF"
VERSION_="5.3"

while getopts "h:b:t:f:c:l:zmsidvq:x:" opt; do
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
    c) CXX_STD=$OPTARG
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
    x) VERSION_=$OPTARG
        ;;
    esac
done

if [[ "$VERSION_"  != "4.9" ]] && [[ "$VERSION_" != "5.3" ]]; then
    echo "VERSION $VERSION_ not supported"
    help
fi
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

if [[ "$CXX_STD" != "cxx11" ]] && [[ "$CXX_STD" != "cxx03" ]]; then
   help
fi

echo $@

source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh
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
    USE_GPU=ON
else
    USE_GPU=OFF
fi
echo "USE_GPU=$USE_GPU"

if [[ "$FLOAT_TYPE" == "float" ]]; then
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

RUN_MPI_TESTS=$USE_MPI ##$SINGLE_PRECISION

pwd
WHERE_=`pwd`

export JENKINS_COMMUNICATION_TESTS=1

if [[ -z ${ICOSAHEDRAL_GRID} ]]; then
    STRUCTURED_GRIDS="ON"
else
    STRUCTURED_GRIDS="OFF"
fi

# measuring time
export START_TIME=$SECONDS

echo "Building on `hostname`"

cmake \
-DBoost_NO_BOOST_CMAKE="true" \
-DCUDA_NVCC_FLAGS:STRING="--relaxed-constexpr" \
-DCUDA_ARCH:STRING="$CUDA_ARCH" \
-DCMAKE_BUILD_TYPE:STRING="$BUILD_TYPE" \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DGPU_ENABLED_FUSION:PATH=../fusion/include \
-DUSE_GPU:BOOL=$USE_GPU \
-DGNU_COVERAGE:BOOL=OFF \
-DGCL_ONLY:BOOL=OFF \
-DCMAKE_CXX_COMPILER="${HOST_COMPILER}" \
-DCMAKE_CXX_FLAGS:STRING="-I${MPI_HOME}/include ${ADDITIONAL_FLAGS}" \
-DCUDA_HOST_COMPILER:STRING="${HOST_COMPILER}" \
-DUSE_MPI:BOOL=$USE_MPI \
-DUSE_MPI_COMPILER:BOOL=$USE_MPI_COMPILER  \
-DSINGLE_PRECISION:BOOL=$SINGLE_PRECISION \
-DENABLE_CXX11:BOOL=$CXX_11 \
-DENABLE_PERFORMANCE_METERS:BOOL=ON \
-DSTRUCTURED_GRIDS:BOOL=${STRUCTURED_GRIDS} \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DVERBOSE=$VERBOSE_RUN \
 ../

exit_if_error $?

#number of trials for compilation. We add this here because sometime intermediate links of nvcc are missing
#some object files, probably related to parallel make compilation, but we dont know yet how to solve this.
#Workaround here is to try multiple times the compilation step
num_make_rep=2

error_code=0
log_file="/tmp/jenkins_${BUILD_TYPE}_${TARGET}_${FLOAT_TYPE}_${CXX_STD}_${MPI}_${RANDOM}.log"
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

queue_str=""
if [[ ${QUEUE} ]] ; then
  queue_str="-q ${QUEUE}"
fi


if [[ "$RUN_MPI_TESTS" == "ON" ]]; then
    bash ${INITPATH}/${BASEPATH_SCRIPT}/test.sh ${queue_str} -m $RUN_MPI_TESTS -n $MPI_NODES -t $MPI_TASKS -g $USE_GPU
else
    bash ${INITPATH}/${BASEPATH_SCRIPT}/test.sh ${queue_str}
fi

exit_if_error $?

if [[ "$RUN_MPI_TESTS" == "ON"  && ${STRUCTURED_GRIDS} == "ON" ]]
then
  if [ "x$CXX_STD" == "xcxx11" ]
  then
      if [ "x$TARGET" == "xcpu" ]
      then
          export MV2_USE_GPUDIRECT_GDRCOPY=0
          export HOST_NAME=`hostname`
          echo "running shallow water test with MPI"
          echo "mpirun -np 4 ./build/shallow_water_enhanced 8 8 1 10"
          mpirun -np 4 ./examples/shallow_water_enhanced 8 8 1 10
          exit_if_error $?

          echo "running copy stencil test with MPI"
          echo "mpirun -np 2 ./build/copy_stencil_parallel 62 53 15"
          mpirun -np 2 ./examples/copy_stencil_parallel 62 53 15
          exit_if_error $?
      fi
  fi
fi

exit 0
