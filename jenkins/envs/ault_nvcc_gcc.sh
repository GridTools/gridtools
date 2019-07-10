#!/bin/bash

source $(dirname "$BASH_SOURCE")/ault.sh

export CXX=$(which g++)
export CC=$(which gcc)
export FC=$(which gfortran)
export GTCMAKE_CMAKE_CUDA_HOST_COMPILER="$CXX"
