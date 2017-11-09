#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh

source ${JENKINSPATH}/env_perftest_${myhost}.sh

TEMP=`getopt -o h --long target:,prec:,jplan:,json:,gtype: \
             -n 'jenkins_perftest' -- "$@"`

eval set -- "$TEMP"

while true; do 
    case "$1" in
        --target) TARGET=$2; shift 2;;
        --prec) PREC=$2; shift 2;;
        --jplan) JPLAN=$2; shift 2;;
        --json) JSON_FILE=$2; shift 2;;
        --gtype) GTYPE=$2; shift 2;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

STD="cxx11"

if [[ -z ${TARGET} || -z ${STD} || -z ${PREC} ]]; then
    echo "Error: some arguments are not set"
    exit 1
fi

if [[ ${JPLAN} != "GridTools" && ${JPLAN} != "GridTools_icgrid" && ${JPLAN} != "GridTools_strgrid_PR" 
    && ${JPLAN} != "GridTools_icgrid_PR" ]]; then
    echo "JENKINS PLAN not set or not supported : ${JPLAN}"
    exit 1
fi

if [[ -z ${JSON_FILE} ]]; then
    echo "--json must be specified"
    exit 1
fi
if [[ -z ${GTYPE} ]]; then
    echo "Grid Type --gtype must be specified"
    exit 1
fi

if [[ ${JPLAN} == "GridTools" ]]; then
  GPATH="${GRIDTOOLS_BUILD_PATH}/${JPLAN}/build_type/release/compiler/gcc/label/${myhost}/mpi/MPI/"
else
  GPATH="${GRIDTOOLS_BUILD_PATH}/${JPLAN}/build_type/release/compiler/gcc/label/${myhost}/mpi/MPI/"
fi

export GPATH=${GPATH}/real_type/$PREC/std/$STD/target/$TARGET/build
export STELLA_PATH=${STELLA_BUILD_PATH}/stella/trunk_timers/release_$PREC/bin/
cd ${JENKINSPATH}/
cmd="python process_ref.py -p $GPATH --target $TARGET --prec $PREC -c -u ${JSON_FILE} --stella_path $STELLA_PATH --gtype ${GTYPE} -v --plot"
echo "$cmd"
$cmd

