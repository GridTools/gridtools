#!/bin/bash

TARGET=$1
REAL_TYPE=$2
if [ "x$TARGET" == "xgpu" ]
then
USE_GPU=ON
PAPI_WRAP_LIBRARY=OFF
else
USE_GPU=OFF
PAPI_WRAP_LIBRARY=ON
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

if [ "x$CXX_11_ON" == "xcxx11" ]
then
CXX_11=ON
else
CXX_11=OFF
fi
echo "C++ 11 = $CXX_11"

module load cmake
module load boost/1.56.0
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
-DGPU_ENABLED_FUSION:PATH=./fusion/include \
-DBoost_INCLUDE_DIR:PATH=/apps/daint/boost/1.56.0/gnu_482/include \
-DBoost_DIR:PATH=/apps/daint/boost/1.56.0/gnu_482  \
-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-DUSE_PAPI:BOOL=OFF \
-DGNU_COVERAGE:BOOL=ON \
-DGCOVR_PATH:PATH=/users/crosetto/gcovr-3.2/scripts \
-DPAPI_WRAP_LIBRARY:BOOL=$PAPI_WRAP_LIBRARY \
-DGCL_ONLY:BOOL=OFF \
-DUSE_MPI:BOOL=ON \
-DUSE_MPI_COMPILER:BOOL=OFF \
-DPAPI_WRAP_PREFIX:PATH=/users/crosetto/builds/GridTools/gridtools/include/external/perfcount \
-DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3 -m64 -mavx -DNDEBUG -DBOOST_SYSTEM_NO_DEPRECATED"  \
-DSINGLE_PRECISION=$SINGLE_PRECISION \
-DENABLE_CXX11=$CXX_11 \
../

make -j8;

if [ "x$TARGET" == "xgpu" ]
then
make tests_gpu;
# echo "#!/bin/bash
# export MODULEPATH=\"/opt/totalview-support/1.1.4/modulefiles:/opt/cray/craype/default/modulefiles:/opt/cray/ari/modulefiles:/opt/cray/modulefiles:/opt/modulefiles:/cm/local/modulefiles:/cm/shared/modulefiles:/apps/daint/modulefiles\"
# module() { eval `/opt/modules/3.2.6.7/bin/modulecmd bash $*`; }

# echo \"loading modules\"
# module load boost
# echo \"modules loaded: start test execution\"

# aprun \"/scratch/daint/jenkins/~/test/real_type/$REAL_TYPE/slave/daint/target/gpu/build/build/tests_gpu\"" > /users/jenkins/runTest_daint.sh
# chmod +x /users/jenkins/runTest_daint.sh
echo "disabled execution on GPUs because of an obscure problem with rpath/dynamic libraries"
#salloc --gres=gpu:1 /users/jenkins/runTest_daint.sh
else
make tests;
./build/tests
fi
rm -rf *
