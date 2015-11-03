#!/bin/bash

CONTAINER_NAME="$( docker ps | grep gridtools4py | awk '{ print $1; }' )"
CONTAINER_IP=$( docker inspect ${CONTAINER_NAME} | grep \"IPAddress | awk '{ print $2; }' | tr -d ',' | tr -d '"' )
EXE_CMD='for v in $( cat ~/.profile ); do export ${v}; done                 && \
         cd $HOME/gridtools/python                                          && \
	 ./python_run_tests.sh                                                    '
#         ipython notebook --ip=0.0.0.0 --port=8888 --no-browser Gridtools4Py.ipynb'
echo ">>> IPython notebook listening at ${CONTAINER_IP} <<<"
ssh -i ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -YC docker@${CONTAINER_IP} "${EXE_CMD}"
