#!/bin/bash

TARGET=$1
REAL_TYPE=$2
if [ "x$TARGET" == "xgpu" ]
then
USE_GPU=ON
else
USE_GPU=OFF
fi
echo "USE_GPU=$USE_GPU"

if [ "x$REAL_TYPE" == "xfloat" ]
then
SINGLE_PRECISION=ON
else
INGLE_PRECISION=OFF
fi
echo "SINGLE_PRECISION=$SINGLE_PRECISION"
pwd

module load cmake
module load boost
module unload  PrgEnv-cray
module load  PrgEnv-gnu
module load cudatoolkit
module load papi
module load gcc/4.8.2
export PAPI_ROOT=/opt/cray/papi/5.2.0
export PAPI_WRAP_ROOT=/users/crosetto/builds/GridTools/gridtools/include/external/perfcount/
export CSCSPERF_EVENTS="SIMD_FP_256|PAPI_VEC_DP|PAPI_VEC_SP"
echo "modules loaded: start compilation"

cmake \
-DCUDA_NVCC_FLAGS:STRING="-arch=sm_30  --ptxas-options -v" \
-DCUDA_SDK_ROOT_DIR:PATH=/opt/nvidia/cudatoolkit/5.5.20-1.0501.7945.8.2 \
-DUSE_GPU:BOOL=$USE_GPU \
-DGTEST_ROOT=/project/csstaff/mbianco/googletest/ \
-DGPU_ENABLED_FUSION:PATH=../fusion/include \
-DBoost_INCLUDE_DIR:PATH=/apps/daint/boost/1.54.0/gnu_473/include \
-DBoost_DIR:PATH=/apps/daint/boost/1.54.0/gnu_473  \
-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-DUSE_PAPI:BOOL=OFF \
-DGNU_COVERAGE:BOOL=ON \
-DGCOVR_PATH:PATH=/users/crosetto/gcovr-3.2/scripts \
-DPAPI_WRAP_LIBRARY:BOOL=ON \
-DGCL_ONLY:BOOL=OFF \
-DUSE_MPI:BOOL=ON \
-DUSE_MPI_COMPILER:BOOL=OFF \
-DPAPI_WRAP_PREFIX:PATH=/users/crosetto/builds/GridTools/gridtools/include/external/perfcount \
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3 -m64 -mavx -DNDEBUG"  \
-DSINGLE_PRECISION=$SINGLE_PRECISION \
../

make -j8;

if [ "x$TARGET" == "xgpu" ]
then
make tests_gpu;
salloc --gres=gpu:1 aprun "/scratch/daint/jenkins/~/test/real_type/float/slave/daint/target/cpu/build/build/tests_gpu"
else
make tests;
./build/tests
fi
rm -rf *
