#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh

source ${JENKINSPATH}/env_perftest_${myhost}.sh

TEMP=`getopt -o h --long target:,std:,prec: \
             -n 'jenkins_perftest' -- "$@"`

eval set -- "$TEMP"

while true; do 
    case "$1" in
        --target) TARGET=$2; shift 2;;
        --std) STD=$2; shift 2;;
        --prec) PREC=$2; shift 2;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

if [[ -z ${TARGET} || -z ${STD} || -z ${PREC} ]]; then
    echo "Error: some arguments are not set"
    exit 1
fi

export GPATH=${GRIDTOOLS_BUILD_PATH}/GridTools/build_type/release/label/${myhost}/mpi/MPI/python/python_off/real_type/$PREC/std/$STD/target/$TARGET/build
export STELLA_PATH=${STELLA_BUILD_PATH}/stella/trunk/release_$PREC/bin/

cd ${JENKINSPATH}/
cmd="python process_ref.py -p $GPATH --target $TARGET --std $STD --prec $PREC -c -u stencils.json --stella_path $STELLA_PATH -v --plot"
$cmd

