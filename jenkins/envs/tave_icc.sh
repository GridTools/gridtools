#!/bin/bash

source $(dirname "$BASH_SOURCE")/tave.sh

module load PrgEnv-intel
module load gcc

export CXX=$(which icpc)
export CC=$(which icc)
export FC=$(which ifort)
export GTCMAKE_CMAKE_CXX_FLAGS='-xmic-avx512'

export KMP_AFFINITY='balanced'
