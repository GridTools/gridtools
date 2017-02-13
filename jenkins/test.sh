#!/bin/bash -f

while getopts "q:" opt; do
    case "$opt" in
    q) QUEUE=$OPTARG
        ;;
    esac
done

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/tools.sh
source ${JENKINSPATH}/machine_env.sh
maxsleep=7200

if [[ -z ${DEFAULT_QUEUE} ]]; then
    exitError 3485 ${LINENO} "Default queue not set"
fi

if [[ -z ${QUEUE} ]]; then
    QUEUE=${DEFAULT_QUEUE}
fi

testfile=${JENKINSPATH}/../build/test.out
test -e ${testfile}
if [ $? -eq 0 ] ; then
    echo Deleting previus test results
    rm ${testfile}
fi

echo source ${JENKINSPATH}/env_${myhost}.sh
source ${JENKINSPATH}/env_${myhost}.sh
cp ${JENKINSPATH}/submit.${myhost}.slurm ${JENKINSPATH}/submit.${myhost}.slurm.test
slurm_script="${JENKINSPATH}/submit.${myhost}.slurm.test"

if [ $myhost == "greina" ]; then
    cmd="srun --gres=gpu:1 --ntasks=1 -u  bash ${JENKINSPATH}/../build/run_tests.sh "
elif [ $myhost == "kesch" ]; then
    cmd="srun --ntasks=1 -K -u bash ${JENKINSPATH}/../build/run_tests.sh"
elif [ $myhost == "daint" ]; then
    cmd="aprun -B bash ${JENKINSPATH}/../build/run_tests.sh"
fi
echo "replacing in ${slurm_script} command by ${cmd}"
/bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}
/bin/sed -i 's|<QUEUE>|'"${QUEUE}"'|g' ${slurm_script}
/bin/sed -i 's|<CPUSPERTASK>|'"1"'|g' ${slurm_script}

bash ${JENKINSPATH}/monitorjobid `sbatch ${slurm_script} | gawk '{print $4}'` $maxsleep

test -e ${testfile}
if [ $? -ne 0 ] ; then
    # abort
    exitError 4652 ${LINENO} "Output of test file not found"
fi

grep -i 'fail\|error\|[^a-zA-z]fault' ${testfile}

if [ $? -eq 0 ] ; then
    # echo output to stdout
    test -f ${testfile} || exitError 6550 ${LINENO} "batch job output file missing"
    echo "=== test.out BEGIN ==="
    cat ${testfile} | /bin/sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"
    echo "=== test.out END ==="
    # abort
    exitError 4654 ${LINENO} "problem with unittests for test data detected"
else
    echo "Unittests successful (see test.out for detailed log)"
fi

# end timer and report time taken
T=$(($SECONDS - $START_TIME))
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0
