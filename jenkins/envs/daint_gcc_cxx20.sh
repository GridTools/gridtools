#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint_nvcc_gcc.sh
module switch gcc/10.3.0
module unload cudatoolkit

export GTCMAKE_GT_TESTS_REQUIRE_GPU=OFF
export GTCMAKE_GT_TESTS_CXX_STANDARD=20

