#!/bin/sh

source tave.sh

module load PrgEnv-gnu
module swap gcc/7.3.0

export CXX=$(which g++)
export CC=$(which gcc)
export FC=$(which gfortran)

export GTCMAKE_CMAKE_CXX_FLAGS='-march=knl -fvect-cost-model=unlimited'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export OMP_PLACES='{0,64}:64'
export OMP_WAIT_POLICY=active
