#!/bin/bash
set -e

#
# set UID from the host
#
usermod -u ${HOST_UID} dev > /dev/null 2>&1

#
# environment setup
# BOOST_ROOT and CUDATOOLKIT_HOME are set when building the container image
#
export CXX=/usr/bin/g++
export GRIDTOOLS_ROOT=$HOME/gridtools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDATOOLKIT_HOME/lib64

#
# start the IPython notebook
#
MY_IP="$( ifconfig | grep -A1 'Ethernet' | grep inet | cut -d':' -f2 | cut -d' ' -f1 )"
echo ">>> Point your browser to $MY_IP:8888 <<<"
cd $HOME/gridtools/python
su -m -c "ipython notebook --ip=0.0.0.0 Gridtools4Py.ipynb" dev
