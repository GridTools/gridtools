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

echo "loading cuda"
module unload cuda
module load cuda/6.5

echo "exporting variables"
export PAPI_ROOT=/opt/cray/papi/5.2.0
export PAPI_WRAP_ROOT=/users/crosetto/builds/GridTools/gridtools/include/external/perfcount/
export CSCSPERF_EVENTS="SIMD_FP_256|PAPI_VEC_DP|PAPI_VEC_SP"module unload gcc
module load gcc/4.8.2


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
SINGLE_PRECISION=OFF
fi
echo "SINGLE_PRECISION=$SINGLE_PRECISION"

pwd
#mkdir build; cd build;

/apps/dom/cmake/repository/gnu_446/bin/cmake \
-DCUDA_NVCC_FLAGS:STRING="-arch=sm_35 -G " \
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
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3  -g  -m64  -DBOOST_SYSTEM_NO_DEPRECATED"  \
-DSINGLE_PRECISION=$SINGLE_PRECISION \
-DENABLE_CXX11=ON \
 ../

make -j8;

if [ "x$TARGET" == "xgpu" ]
then
make tests_gpu;
salloc --gres=gpu:1 srun "/scratch/shared/castor/jenkins/dom/~/test/real_type/$REAL_TYPE/slave/dom/target/gpu/build/tests_gpu"
else
make tests;
/scratch/shared/castor/jenkins/dom/~/test/real_type/$REAL_TYPE/slave/dom/target/$TARGET/build/tests
fi
rm -rf *
