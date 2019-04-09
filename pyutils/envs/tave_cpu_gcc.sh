#!/bin/sh

source tave.sh

module load PrgEnv-gnu
module swap gcc/7.3.0

export GTCMAKE_CMAKE_CXX_COMPILER=$(which CC)
export GTCMAKE_CMAKE_C_COMPILER=$(which cc)
export GTCMAKE_CMAKE_FORTRAN_COMPILER=$(which ftn)
