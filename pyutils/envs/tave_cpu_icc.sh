#!/bin/sh

source tave.sh

module load PrgEnv-intel

export GTCMAKE_CMAKE_CXX_COMPILER=$(which CC)
export GTCMAKE_CMAKE_C_COMPILER=$(which cc)
export GTCMAKE_CMAKE_FORTRAN_COMPILER=$(which ftn)
