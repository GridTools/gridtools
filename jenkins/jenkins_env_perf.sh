#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh

source ${JENKINSPATH}/jenkins_perftest_${myhost}.sh


module load matplotlib/1.4.3-foss-2015a-Python-2.7.9
