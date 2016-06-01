#!/bin/bash

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh
source ${JENKINSPATH}/env_perftest_${myhost}.sh
source ${JENKINSPATH}/tools.sh
echo ${JENKINSPATH}

TEMP=`getopt -o h --long target:,std:,prec:,jplan:,python: \
             -n 'jenkins_perftest' -- "$@"`

eval set -- "$TEMP"

while true; do 
    case "$1" in
        --target) TARGET=$2; shift 2;;
        --std) STD=$2; shift 2;;
        --prec) PREC=$2; shift 2;;
        --jplan) JPLAN=$2; shift 2;;
        --python) PYTHON_OPT=$2; shift 2;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

if [[ -z ${TARGET} || -z ${STD} || -z ${PREC} ]]; then
    echo "Error: some arguments are not set"
    exit 1
fi
maxsleep=7200

if [[ -n "${PYTHON_OPT}" ]]; then
    PYTHON_STR="--python ${PYTHON_OPT}"
fi

if [ "$myhost" == "greina" ]; then
    bash ${JENKINSPATH}/jenkins_perftest_exec.sh --target $TARGET --std $STD --prec $PREC ${PYTHON_STR} --jplan $JPLAN 
else

    cp ${JENKINSPATH}/submit.kesch.slurm ${JENKINSPATH}/submit.kesch.slurm.test
    slurm_script="${JENKINSPATH}/submit.kesch.slurm.test"
    cmd="srun --gres=gpu:1 --ntasks=1 -u bash ${JENKINSPATH}/jenkins_perftest_exec.sh --target $TARGET --std $STD --prec $PREC ${PYTHON_STR} --jplan $JPLAN"
    /bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}

    bash ${JENKINSPATH}/monitorjobid `sbatch ${slurm_script} | gawk '{print $4}'` $maxsleep
    grep -E 'Error in conf|FAILED|ERROR' test.out 


    if [ $? -eq 0 ] ; then
        # echo output to stdout
        test -f test.out || exitError 6550 ${LINENO} "batch job output file missing"
        echo "=== test.out BEGIN ==="
        cat test.out | /bin/sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"
        echo "=== test.out END ==="
        # abort
        exitError 4654 ${LINENO} "problem with unittests for test data detected"
    else
        echo "Perftests successful (see test.out for detailed log)"
    fi
fi

