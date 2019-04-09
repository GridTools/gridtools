#!/bin/sh

source daint.sh

module load /users/vogtha/modules/compilers/clang/7.0.1

export GTCMAKE_CXX_COMPILER=$(which clang++)
