#!/bin/sh

source base.sh

module rm PrgEnv-cray
module rm CMake
module load /users/jenkins/easybuild/tave/modules/all/CMake/3.12.4

export BOOST_ROOT=/project/c14/install/kesch/boost/boost_1_67_0 #since it is header only we can use the kesch installation
export GTCMAKE_GT_ENABLE_BINDINGS_GENERATION=OFF
export GTCMAKE_MPITEST_PREFLAGS='numactl;-m;1'

export GTRUN_BUILD_COMMAND='make -j 16'
export GTRUN_SBATCH_CONSTRAINT='flat,quad'
export GTRUN_SBATCH_NODES=1
export GTRUN_SBATCH_NTASKS_PER_CORE=4
export GTRUN_SBATCH_NTASKS_PER_NODE=1
export GTRUN_SBATCH_CPUS_PER_TASK=256
export GTRUN_SBATCH_TIME='00:15:00'
export GTRUNMPI_SBATCH_NODES=4

export OMP_NUM_THREADS=128
