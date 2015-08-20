#!/bin/bash
set -e

#
# set UID from the host
#
usermod -u ${HOST_UID} dev > /dev/null 2>&1

#
# environment setup
# BOOST_ROOT and CUDATOOLKIT_HOME are set when building the image
#
export CXX=/usr/bin/g++
export GRIDTOOLS_ROOT=$HOME/gridtools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDATOOLKIT_HOME/lib64


#
# run some tests
#
#cd $HOME/gridtools/python
#nosetests -v -s -x tests.test_stencils

EXE_CMD="$@"
su -m -c "${EXE_CMD}" dev
