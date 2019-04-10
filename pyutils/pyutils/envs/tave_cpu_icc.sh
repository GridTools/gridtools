#!/bin/sh

source tave.sh

module load PrgEnv-intel

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)

export KMP_AFFINITY='balanced'
