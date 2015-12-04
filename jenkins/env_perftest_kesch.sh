#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh

module load matplotlib/1.4.3-gmvolf-15.11-Python-2.7.10

export GRIDTOOLS_BUILD_PATH=/scratch/jenkins/workspace/
export STELLA_BUILD_PATH=/project/c01/install/${myhost}
