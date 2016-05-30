#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh

export GRIDTOOLS_BUILD_PATH=/home/jenkins/workspace/
export STELLA_BUILD_PATH=/users/jenkins/install/${myhost}
