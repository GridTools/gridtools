#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint_nvcc_gcc.sh

export GTCMAKE_GT_TESTS_CXX_STANDARD=17
