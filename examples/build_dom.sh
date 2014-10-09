#!/bin/bash
. /apps/dom/Modules/3.2.10/init/bash
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
#mkdir build; cd build;

/apps/dom/cmake/repository/gnu_446/bin/cmake \
-DCUDA_NVCC_FLAGS:STRING="-arch=sm_35  -G -std=c++11 --ptxas-options -v " \
-DCUDA_SDK_ROOT_DIR:PATH=/opt/nvidia/cudatoolkit/5.5.20-1.0501.7945.8.2 \
-DCMAKE_CXX_COMPILER:STRING="/apps/dom/gcc/4.8.2/bin/c++" \
-DCMAKE_C_COMPILER:STRING="/apps/dom/gcc/4.8.2/bin/gcc" \
-DUSE_GPU:BOOL=$USE_GPU \
-DGTEST_ROOT=/project/csstaff/mbianco/googletest/ \
-DGPU_ENABLED_FUSION:PATH=../fusion/include \
-DBoost_DIR:PATH=/users/mbianco/boost_1_55_0 \
-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-DUSE_PAPI:BOOL=OFF \
-DGNU_COVERAGE:BOOL=ON \
-DGCOVR_PATH:PATH=/users/crosetto/gcovr-3.2/scripts \
-DPAPI_WRAP_LIBRARY:BOOL=OFF \
-DGCL_ONLY:BOOL=OFF \
-DUSE_MPI:BOOL=OFF \
-DUSE_MPI_COMPILER:BOOL=OFF  \
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3  -g -std=c++11 -m64"  \
 ../

make -j8;

if [ "x$TARGET" == "xgpu" ]
then
make tests_gpu;
salloc --gres=gpu:1 aprun "/scratch/shared/castor/jenkins/dom/~/test/real_type/$REAL_TYPE/slave/dom/target/gpu/build/build/tests_gpu"
else
make tests;
./build/tests
fi
rm -rf *
ls /bin
