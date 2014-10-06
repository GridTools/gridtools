#!/bin/bash

module load cmake
module load boost
module load papi
module unload cuda
module load cuda/6.5
export PAPI_ROOT=/opt/cray/papi/5.2.0
export PAPI_WRAP_ROOT=/users/crosetto/builds/GridTools/gridtools/include/external/perfcount/
export CSCSPERF_EVENTS="SIMD_FP_256|PAPI_VEC_DP|PAPI_VEC_SP"module unload gcc
module load gcc/4.8.2

pwd
ls
mkdir build; cd build; 

cmake \
-DCUDA_NVCC_FLAGS:STRING=-arch=sm_35 \
-DCUDA_SDK_ROOT_DIR:PATH=/opt/nvidia/cudatoolkit/5.5.20-1.0501.7945.8.2 \
-DUSE_GPU:BOOL=ON \
-DGPU_ENABLED_FUSION:PATH=../fusion/include \
-DBoost_INCLUDE_DIR:PATH=/apps/daint/boost/1.54.0/gnu_473/include \
-DBoost_DIR:PATH=/apps/daint/boost/1.54.0/gnu_473  \
-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-DUSE_PAPI:BOOL=ON \
-DGNU_COVERAGE:BOOL=ON \
-DPAPI_PREFIX:PATH=/opt/cray/papi/5.2.0 \
-DPAPI_WRAP_LIBRARY:BOOL=ON \
-DGCL_ONLY:BOOL=OFF \
-DUSE_MPI:BOOL=ON \
-DUSE_MPI_COMPILER:BOOL=OFF  \
-DPAPI_WRAP_PREFIX:PATH=~/builds/GridTools/gridtools/include/external/perfcount \
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3 -m64 -mavx -DNDEBUG -DUSE_PAPI_WRAP"  \
 ../

make -j8; make test; rm -rf *