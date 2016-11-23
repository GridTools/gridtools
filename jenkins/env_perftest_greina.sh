#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh
module load slurm
export GRIDTOOLS_BUILD_PATH=/home/jenkins/workspace/
export STELLA_BUILD_PATH=/users/jenkins/install/${myhost}
export DEFAULT_QUEUE=k40
export CPUS_PER_SOCKET=8
