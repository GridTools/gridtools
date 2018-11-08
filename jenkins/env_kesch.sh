#/bin/bash

module load craype-network-infiniband
module load craype-haswell
module load craype-accel-nvidia35
module load cray-libsci
module load cudatoolkit/8.0.61
module load mvapich2gdr_gnu/2.2_cuda_8.0
module load gcc/5.4.0-2.26
module load /users/mlukas/easybuild/kesch/modules/all/cmake/3.12.4

export HOST_COMPILER=`which g++`
export Boost_NO_SYSTEM_PATHS=true
export Boost_NO_BOOST_CMAKE=true
export GRIDTOOLS_ROOT_BUILD=$PWD/build
export GRIDTOOLS_ROOT=$PWD
export CUDATOOLKIT_HOME=${CUDA_PATH}
export BOOST_ROOT=/project/c14/install/kesch/boost/boost_1_67_0
export BOOST_INCLUDE=/project/c14/install/kesch/boost/boost_1_67_0/include/
export CUDA_ARCH=sm_37
export DEFAULT_QUEUE=debug
export LAUNCH_MPI_TEST="srun"

export JOB_ENV_COMMON="export CUDA_AUTO_BOOST=0;export GCLOCK=875; export LD_PRELOAD=/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5/lib64/libmpi.so;export OMP_NUM_THREADS=1;export MALLOC_MMAP_MAX_=0;export MALLOC_TRIM_THRESHOLD_=536870912"
export JOB_ENV="${JOB_ENV_COMMON}; export G2G=1"
export MPI_HOST_JOB_ENV=""
export MPI_CUDA_JOB_ENV="${JOB_ENV_COMMON}; export G2G=2; export MV2_USE_GPUDIRECT=0; export MV2_USE_RDMA_FAST_PATH=0"
export MPI_NODES=1
export MPI_TASKS=4
export CXX=`which g++`
export CC=`which gcc`
export FC=`which gfortran`
export MAKE_THREADS=12
export SRUN_BUILD_COMMAND="srun -p pp-short -c 12 --time=00:30:00"

