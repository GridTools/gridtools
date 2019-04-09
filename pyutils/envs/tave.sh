#!/bin/sh

module rm PrgEnv-cray
module rm CMake
module load /users/jenkins/easybuild/tave/modules/all/CMake/3.12.4

export BOOST_ROOT=/project/c14/install/kesch/boost/boost_1_67_0 #since it is header only we can use the kesch installation
export GTCMAKE_GT_ENABLE_BINDINGS_GENERATION=OFF

export GTCI_QUEUE=normal
export GTCI_MPI_NODES=4
export GTCI_MPI_TASKS=4
export GTCI_BUILD_THREADS=16
export GTCI_BUILD_COMMAND=""
