#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint_cray.sh

module switch cce/12.0.2

export GCC_X86_64=/opt/gcc/10.3.0/snos
export GTCMAKE_GT_TESTS_CXX_STANDARD=20
# some problem with C++20 math library included by Python (std::lerp not defined)
export GTCMAKE_GT_TESTS_ENABLE_PYTHON_TESTS=OFF
