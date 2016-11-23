#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh
source ${JENKINSPATH}/env_perftest_${myhost}.sh
source ${JENKINSPATH}/tools.sh
echo ${JENKINSPATH}

TEMP=`getopt -o h --long target:,std:,prec:,jplan:,python:,outfile:,json:,gtype: \
             -n 'jenkins_perftest' -- "$@"`

eval set -- "$TEMP"

while true; do 
    case "$1" in
        --target) TARGET=$2; shift 2;;
        --std) STD=$2; shift 2;;
        --prec) PREC=$2; shift 2;;
        --jplan) JPLAN=$2; shift 2;;
        --python) PYTHON_OPT=$2; shift 2;;
        --outfile) OUTFILE=$2; shift 2;;
        --json) JSON_FILE=$2; shift 2;;
        --gtype) GTYPE=$2; shift 2;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

if [[ -z ${DEFAULT_QUEUE} ]]; then
    echo "Error: default queue not defined" 
    exit 1
fi
QUEUE=${DEFAULT_QUEUE}
#setting default compiler to gcc
export COMPILER="gcc"

if [[ -z ${TARGET} || -z ${STD} || -z ${PREC} ]]; then
    echo "Error: some arguments are not set"
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
maxsleep=7200

if [[ -n "${PYTHON_OPT}" ]]; then
    PYTHON_STR="--python ${PYTHON_OPT}"
fi

slurm_script="${JENKINSPATH}/submit.${myhost}.slurm.test.${RANDOM}"
cp ${JENKINSPATH}/submit.${myhost}.slurm ${slurm_script}
cmd="srun --gres=gpu:1 --ntasks=1 -u bash ${JENKINSPATH}/jenkins_perftest_exec.sh --target $TARGET --std $STD --prec $PREC ${PYTHON_STR} --jplan $JPLAN --json ${JSON_FILE} --gtype ${GTYPE}"
/bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}
/bin/sed -i 's|<QUEUE>|'"${QUEUE}"'|g' ${slurm_script}

if [[ ${TARGET} == "cpu" ]]; then
    if [[ -z ${CPUS_PER_SOCKET} ]]; then
        echo "CPUS_PER_SOCKET not defined"
        exit 1
    fi
    /bin/sed -i 's|<CPUSPERTASK>|'"${CPUS_PER_SOCKET}"'|g' ${slurm_script}
else
    /bin/sed -i 's|<CPUSPERTASK>|'"1"'|g' ${slurm_script}
fi

if [[ -n ${OUTFILE} ]]; then
    /bin/sed -i 's|test.out|'"${OUTFILE}"'|g' ${slurm_script}
else
    OUTFILE=test.out
fi


export CUDA_AUTO_BOOST=0; export GCLOCK=875;

bash ${JENKINSPATH}/monitorjobid `export CUDA_AUTO_BOOST=0; export GCLOCK=875; sbatch ${slurm_script} | gawk '{print $4}'` $maxsleep

rm ${slurm_script}

grep -E 'Error in conf|FAILED|ERROR' ${OUTFILE}
if [ $? -eq 0 ] ; then
    # echo output to stdout
    test -f ${OUTFILE} || exitError 6550 ${LINENO} "batch job output file missing"
    echo "=== ${OUTFILE} BEGIN ==="
    cat ${OUTFILE} | /bin/sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"
    echo "=== ${OUTFILE} END ==="
    # abort
    exitError 4654 ${LINENO} "problem with unittests for test data detected"
else
    echo "Perftests successful (see ${OUTFILE} for detailed log)"
fi

