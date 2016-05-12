#!/bin/bash -f

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/tools.sh
source ${JENKINSPATH}/machine_env.sh
maxsleep=7200

test -e test.out
if [ $? -eq 0 ] ; then
    echo Deleting previus test results
    rm test.out
fi

echo source ${JENKINSPATH}/env_${myhost}.sh
source ${JENKINSPATH}/env_${myhost}.sh
cp ${JENKINSPATH}/submit.${myhost}.slurm ${JENKINSPATH}/submit.${myhost}.slurm.test
slurm_script="${JENKINSPATH}/submit.${myhost}.slurm.test"

if [ $myhost == "greina" ]; then
    cmd="srun --gres=gpu:1 --ntasks=1 -u  bash ./run_tests.sh "
elif [ $myhost == "kesch" ]; then
    cmd="srun --ntasks=1 -K -u bash ./run_tests.sh"
elif [ $myhost == "daint" ]; then
    cmd="aprun -B bash ./run_tests.sh"
fi

echo "replacing in ${slurm_script} command by ${cmd}"
/bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}

bash ${JENKINSPATH}/monitorjobid `sbatch ${slurm_script} | gawk '{print $4}'` $maxsleep

test -e test.out
if [ $? -ne 0 ] ; then
    # abort
    exitError 4652 ${LINENO} "Output of test file not found"
fi

grep 'FAILED\|ERROR' test.out

if [ $? -eq 0 ] ; then
    # echo output to stdout
    test -f test.out || exitError 6550 ${LINENO} "batch job output file missing"
    echo "=== test.out BEGIN ==="
    cat test.out | /bin/sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"
    echo "=== test.out END ==="
    # abort
    exitError 4654 ${LINENO} "problem with unittests for test data detected"
else
    echo "Unittests successful (see test.out for detailed log)"
fi

# end timer and report time taken
T="$(($(date +%s)-T))"
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0
