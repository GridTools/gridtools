#!/bin/sh

source daint.sh

module load PrgEnv-gnu

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)
