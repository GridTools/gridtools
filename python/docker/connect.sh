#!/bin/bash

CONTAINER_NAME="$( docker ps | grep gridtools4py | awk '{ print $1; }' )"
CONTAINER_IP=$( docker inspect ${CONTAINER_NAME} | grep \"IPAddress | awk '{ print $2; }' | tr -d ',' | tr -d '"' )
EXE_CMD='export BOOST_ROOT=/usr/local/boost-1.56.0      && \
         export CUDATOOLKIT_HOME=/usr/local/cuda-7.0    && \
         export CXX=/usr/bin/g++                        && \
         export GRIDTOOLS_ROOT=$HOME/gridtools          && \
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDATOOLKIT_HOME/lib64 && \
         cd $HOME/gridtools/python                      && \
         ipython notebook --ip=0.0.0.0 --port=8888 --no-browser Gridtools4Py.ipynb'
echo ">>> IPython notebook listening at ${CONTAINER_IP} <<<"
ssh -i ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -YC dev@${CONTAINER_IP} -p 22 "${EXE_CMD}"
