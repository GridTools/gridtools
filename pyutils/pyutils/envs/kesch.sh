#!/bin/sh

source base.sh

module load craype-network-infiniband
module load craype-haswell
module load craype-accel-nvidia35
module load cray-libsci
module load cudatoolkit/8.0.61
module load mvapich2gdr_gnu/2.2_cuda_8.0
module load gcc/5.4.0-2.26
module load /users/jenkins/easybuild/kesch/modules/all/cmake/3.12.4

export BOOST_ROOT=/project/c14/install/kesch/boost/boost_1_67_0
export CUDATOOLKIT_HOME=$CUDA_PATH
export CUDA_ARCH=sm_37
export LD_PRELOAD=/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5/lib64/libmpi.so

export GTRUN_BUILD_COMMAND='srun -p pp-short -c 12 --time=00:30:00 make -j 12'
export GTRUN_SBATCH_PARTITION='debug'
export GTRUN_SBATCH_NODES=1
export GTRUN_SBATCH_GRES='gpu:1'
export GTRUN_SBATCH_CPUS_PER_TASK=12
export GTRUNMPI_SBATCH_NTASKS=4
export GTRUNMPI_SBATCH_NTASKS_PER_NODE=4
export GTRUNMPI_SBATCH_GRES='gpu:4'

export CUDA_AUTO_BOOST=0
export GCLOCK=875
export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=536870912
export OMP_PROC_BIND='true'
export OMP_NUM_THREADS=12

export CXX=$(which g++)
export CC=$(which gcc)
export FC=$(which n)
