#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

module load cmake/3.21.3
module load boost

export GTRUN_BUILD_COMMAND='make -j 8'
export GTRUN_SBATCH_NODES=1
