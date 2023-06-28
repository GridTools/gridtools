#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

module load cmake
module load boost

export GTRUN_BUILD_COMMAND='make -j 8'
export GTRUN_SBATCH_NODES=1
