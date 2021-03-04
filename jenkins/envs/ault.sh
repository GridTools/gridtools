#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

source /users/fthaler/public/jenkins/spack/share/spack/setup-env.sh

spack load boost
module load cmake/3.18.2

export GTRUN_BUILD_COMMAND='make -j 8'
export GTRUN_SBATCH_NODES=1
