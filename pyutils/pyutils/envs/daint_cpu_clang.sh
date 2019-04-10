#!/bin/sh

source daint.sh

module load /users/vogtha/modules/compilers/clang/7.0.1

export CXX=$(which clang++)
export CC=$(which clang)
