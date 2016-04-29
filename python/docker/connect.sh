#!/bin/bash

CONTAINER_ID="$( docker ps | grep gridtools4py | awk '{ print $1; }' )"
CONTAINER_IP="$( docker inspect ${CONTAINER_ID} | grep '\"IPAddress\"' | awk '{ print $2; }' | head -n 1 | tr -d ',' | tr -d '"' )"
EXE_CMD='for var in $( cat ~/.profile ); do export ${var}; done             && \
         . $HOME/venv/bin/activate                                          && \
         cd $GRIDTOOLS_ROOT/python                                          && \
         ipython notebook --ip=0.0.0.0 --port=8888 --no-browser Gridtools4Py.ipynb'
echo ">>> IPython notebook listening at ${CONTAINER_IP} <<<"
ssh -XC -i ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null docker@${CONTAINER_IP} "${EXE_CMD}"
