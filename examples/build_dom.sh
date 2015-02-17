#!/bin/bash
export MODULEPATH="/apps/dom/Modules/versions:/apps/dom/Modules/$MODULE_VERSION/modulefiles:/apps/dom/Modules/modulefiles:/apps/dom/modulefiles"
module() { eval `/apps/dom/Modules/3.2.10/bin/modulecmd bash $*`; }
#/apps/dom/Modules/3.2.10/bin/modulecmd bash avail
echo "loading cmake"
module load cmake
echo "loading boost"
module load boost
#echo "loading PAPI"
#module load papi

module unload gcc
module load gcc/4.8.2
echo "loading MPI"
module unload mvapich2
module load mvapich2/1.9-gcc-4.8.2
echo "loading cuda"
module unload cuda
module load cuda/6.5

echo "exporting variables"
export PAPI_ROOT=/opt/cray/papi/5.2.0
export PAPI_WRAP_ROOT=/users/crosetto/builds/GridTools/gridtools/include/external/perfcount/
export CSCSPERF_EVENTS="SIMD_FP_256|PAPI_VEC_DP|PAPI_VEC_SP"


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

RUN_MPI_TESTS=$SINGLE_PRECISION

pwd
WHERE_=`pwd`
#mkdir build; cd build;

export JENKINS_COMMUNICATION_TESTS=1
module load cuda/6.0; # load runtime libs

/apps/dom/cmake/repository/gnu_446/bin/cmake \
-DCUDA_NVCC_FLAGS:STRING="-arch=sm_30 -G " \
-DCMAKE_CXX_COMPILER:STRING="/apps/dom/gcc/4.8.2/bin/c++" \
-DCMAKE_C_COMPILER:STRING="/apps/dom/gcc/4.8.2/bin/gcc" \
-DUSE_GPU:BOOL=$USE_GPU \
-DGTEST_ROOT=/project/csstaff/mbianco/googletest/ \
-DGPU_ENABLED_FUSION:PATH=../fusion/include \
-DBoost_DIR:PATH=/users/mbianco/boost_1_55_0 \
-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-DUSE_PAPI:BOOL=OFF \
-DGNU_COVERAGE:BOOL=OFF \
-DGCOVR_PATH:PATH=/users/crosetto/gcovr-3.2/scripts \
-DPAPI_WRAP_LIBRARY:BOOL=OFF \
-DGCL_ONLY:BOOL=OFF \
-DUSE_MPI:BOOL=$RUN_MPI_TESTS \
-DUSE_MPI_COMPILER:BOOL=$RUN_MPI_TESTS  \
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3  -g  -m64  -DBOOST_SYSTEM_NO_DEPRECATED"  \
-DSINGLE_PRECISION=$SINGLE_PRECISION \
-DENABLE_CXX11=ON \
 ../

make -j8;

if [ "x$TARGET" == "xgpu" ]
then
make tests_gpu;
module unload cuda
module load cuda/6.0; # load runtime libs

salloc --gres=gpu:2 srun ./build/tests_gpu

  if [ "$RUN_MPI_TESTS" == "ON" ]
  then
    salloc --gres=gpu:2 ../examples/communication/run_communication_tests.sh
  fi

else
make tests;
module unload cuda
module load cuda/6.0; # load runtime libs
salloc srun ./build/tests

  if [ "$RUN_MPI_TESTS" == "ON" ]
  then
    salloc --gres=gpu:2 ../examples/communication/run_communication_tests.sh
  fi

fi
rm -rf *
