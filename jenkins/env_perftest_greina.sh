#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh

module load matplotlib/1.4.3-foss-2015a-Python-2.7.9

export GRIDTOOLS_BUILD_PATH=/home/jenkins/workspace/
export STELLA_BUILD_PATH=/users/jenkins/install/${myhost}
