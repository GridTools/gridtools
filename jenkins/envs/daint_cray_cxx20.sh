#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint_cray.sh

export GCC_X86_64=/opt/gcc/11.2.0/snos
export GTCMAKE_GT_TESTS_CXX_STANDARD=20

