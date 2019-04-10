#!/bin/sh

source daint.sh

module load PrgEnv-gnu
module swap gcc/7.3.0

export GTCMAKE_CXX=$(which CC)
export GTCMAKE_C_COMPILER=$(which cc)
export GTCMAKE_FORTRAN_COMPILER=$(which ftn)
