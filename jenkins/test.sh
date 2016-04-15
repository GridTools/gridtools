#!/bin/bash -f

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/machine_env.sh
maxsleep=7200

if [ $myhost == "greina" ]; then
    bash ./run_tests.sh
    exit $?
elif [ $myhost == "kesch" ]; then
    source ${JENKINSPATH}/slurmTools.sh
    source ${JENKINSPATH}/env_${myhost}.sh

    cp ${JENKINSPATH}/submit.kesch.slurm ${JENKINSPATH}/submit.kesch.slurm.test
    slurm_script="${JENKINSPATH}/submit.kesch.slurm.test"
    cmd="srun --ntasks=1 -K -u bash ./run_tests.sh"
    echo "replacing in ${slurm_script} command by ${cmd}"
    /bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}

    ~mbianco/bin/monitorjobid `sbatch ${slurm_script} | gawk '{print $4}'`

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
    echo "Unittests successfull (see test.out for detailed log)"
  fi
elif [ $myhost == "daint" ]; then
    source ${JENKINSPATH}/slurmTools.sh
    source ${JENKINSPATH}/env_${myhost}.sh

    cp ${JENKINSPATH}/submit.daint.slurm ${JENKINSPATH}/submit.daint.slurm.test
    slurm_script="${JENKINSPATH}/submit.daint.slurm.test"
    cmd="aprun -B bash ./run_tests.sh"
    echo "replacing in ${slurm_script} command by ${cmd}"
    /bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}

    ~mbianco/bin/monitorjobid `sbatch ${slurm_script} | gawk '{print $4}'`

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
    echo "Unittests successfull (see test.out for detailed log)"
  fi
fi

# end timer and report time taken
T="$(($(date +%s)-T))"
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0
