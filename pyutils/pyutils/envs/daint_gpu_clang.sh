#!/bin/sh

source daint.sh

module load /users/vogtha/modules/compilers/clang/3.8.1

export CXX=$(which clang++)
export CC=$(which clang)
