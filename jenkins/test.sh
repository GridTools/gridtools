#!/bin/bash -f

MPI_NODES=1
MPI_TASKS=1
DO_MPI=OFF
while getopts "q:m:g:n:t:s:" opt; do
    case "$opt" in
    q) QUEUE=$OPTARG
        ;;
    m) DO_MPI=$OPTARG
        ;;
    g) DO_GPU=$OPTARG
        ;;
    n) MPI_NODES=$OPTARG
        ;;
    t) MPI_TASKS=$OPTARG
        ;;
    s) TEST_SCRIPT=$OPTARG
    esac
done

JENKINSPATH=${0%/*}
source ${JENKINSPATH}/tools.sh
source ${JENKINSPATH}/machine_env.sh
maxsleep=7200

if [[ -z ${TEST_SCRIPT} ]]; then
    TEST_SCRIPT="bash ${JENKINSPATH}/../build/run_tests.sh"
fi

if [[ -z ${DEFAULT_QUEUE} ]]; then
    exitError 3485 ${LINENO} "Default queue not set"
fi

if [[ -z ${QUEUE} ]]; then
    QUEUE=${DEFAULT_QUEUE}
fi

testfile=${JENKINSPATH}/../build/test.out
test -e ${testfile}
if [ $? -eq 0 ] ; then
    echo Deleting previous test results
    rm ${testfile}
fi

echo source ${JENKINSPATH}/env_${myhost}.sh
source ${JENKINSPATH}/env_${myhost}.sh
cp ${JENKINSPATH}/submit.${myhost}.slurm ${JENKINSPATH}/submit.${myhost}.slurm.test
slurm_script="${JENKINSPATH}/submit.${myhost}.slurm.test"

if [ $myhost == "greina" ]; then
    cmd="srun --gres=gpu:1 --ntasks=1 -u  ${TEST_SCRIPT} "
elif [ $myhost == "kesch" ]; then
    cmd="srun --ntasks=1 -K -u ${TEST_SCRIPT}"
elif [ $myhost == "dom" ]; then
    cmd="srun ${TEST_SCRIPT}"
elif [ $myhost == "daint" ]; then
    cmd="srun ${TEST_SCRIPT}"
elif [ $myhost == "tave" ]; then
    cmd="srun ${TEST_SCRIPT}"
fi
echo "replacing in ${slurm_script} command by ${cmd}"
/bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}
/bin/sed -i 's|<QUEUE>|'"${QUEUE}"'|g' ${slurm_script}
/bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}
/bin/sed -i 's|<MPI_NODES>|'"1"'|g' ${slurm_script}
/bin/sed -i 's|<MPI_TASKS>|'"1"'|g' ${slurm_script}
/bin/sed -i 's|<MPI_PPN>|'"1"'|g' ${slurm_script}
/bin/sed -i 's|<CPUSPERTASK>|'"1"'|g' ${slurm_script}
/bin/sed -i 's|<OUTPUTFILE>|'"$testfile"'|g' ${slurm_script}
if [ "${JOB_ENV[*]}" == "" ]; then
    /bin/sed -i "s|<JOB_ENV>||g" ${slurm_script}
else
    /bin/sed -i "s|<JOB_ENV>|export ${JOB_ENV[*]}|g" ${slurm_script}
fi

bash ${JENKINSPATH}/monitorjobid `sbatch ${slurm_script} | gawk '{print $4}'` $maxsleep

test -e ${testfile}
if [ $? -ne 0 ] ; then
    # abort
    exitError 4652 ${LINENO} "Output of test file not found"
fi

# grep for failure patterns (exclude ctest summary line)
grep -i 'fail\|error\|[^a-zA-z]fault' ${testfile} | grep -v '100\% tests passed'

if [ $? -eq 0 ] ; then
    # echo output to stdout
    test -f ${testfile} || exitError 6550 ${LINENO} "batch job output file missing"
    echo "=== test.out BEGIN ==="
    cat ${testfile} | /bin/sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"
    echo "=== test.out END ==="
    # abort
    exitError 4654 ${LINENO} "problem with unittests for test data detected"
else
    echo "Unittests successful (see $testfile for detailed log)"
fi

if [[ "$DO_MPI" == "ON" ]]; then
    testfile=${JENKINSPATH}/../build/test$MPI_TASKS\.out
    test -e ${testfile}
    if [ $? -eq 0 ] ; then
        echo Deleting previous test results
        rm ${testfile}
    fi

    cp ${JENKINSPATH}/submit.${myhost}.slurm ${JENKINSPATH}/submit.${myhost}.slurm.mpi.test
    slurm_script="${JENKINSPATH}/submit.${myhost}.slurm.mpi.test"

    cmd="source ${JENKINSPATH}/../build/run_mpi_tests.sh "

    echo "replacing in ${slurm_script} command by ${cmd}"
    /bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}
    /bin/sed -i 's|<QUEUE>|'"${QUEUE}"'|g' ${slurm_script}
    /bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}
    /bin/sed -i 's|<MPI_NODES>|'"${MPI_NODES}"'|g' ${slurm_script}
    /bin/sed -i 's|<MPI_TASKS>|'"$MPI_TASKS"'|g' ${slurm_script}
    PPN=$[MPI_TASKS/MPI_NODES]
    /bin/sed -i 's|<MPI_PPN>|'"$PPN"'|g' ${slurm_script}
    /bin/sed -i 's|<CPUSPERTASK>|'"1"'|g' ${slurm_script}
    /bin/sed -i 's|<OUTPUTFILE>|'"$testfile"'|g' ${slurm_script}
    if [ "${MPI_HOST_JOB_ENV[*]}" == "" ]; then
        /bin/sed -i "s|<JOB_ENV>||g" ${slurm_script}
    else
        /bin/sed -i "s|<JOB_ENV>|export ${MPI_HOST_JOB_ENV[*]}|g" ${slurm_script}
    fi

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
        echo "Unittests successful MPI (see $testfile for detailed log)"
    fi
fi

if [[ $DO_MPI == "ON" && $DO_GPU == "ON" ]]; then
    testfile=${JENKINSPATH}/../build/gputest$MPI_TASKS\.out
    test -e ${testfile}
    if [ $? -eq 0 ] ; then
        echo Deleting previous test results
        rm ${testfile}
    fi

    cp ${JENKINSPATH}/submit.${myhost}.slurm ${JENKINSPATH}/submit.${myhost}.slurm.cuda.mpi.test
    slurm_script="${JENKINSPATH}/submit.${myhost}.slurm.cuda.mpi.test"

    cmd="source ${JENKINSPATH}/../build/run_cuda_mpi_tests.sh "

    echo "replacing in ${slurm_script} command by ${cmd}"
    /bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}
    /bin/sed -i 's|<QUEUE>|'"${QUEUE}"'|g' ${slurm_script}
    /bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm_script}
    /bin/sed -i 's|<MPI_NODES>|'"${MPI_NODES}"'|g' ${slurm_script}
    /bin/sed -i 's|<MPI_TASKS>|'"$MPI_TASKS"'|g' ${slurm_script}
    PPN=$[MPI_TASKS/MPI_NODES]
    /bin/sed -i 's|<MPI_PPN>|'"$PPN"'|g' ${slurm_script}
    /bin/sed -i 's|<CPUSPERTASK>|'"1"'|g' ${slurm_script}
    /bin/sed -i 's|<OUTPUTFILE>|'"$testfile"'|g' ${slurm_script}
    if [ "${MPI_CUDA_JOB_ENV[*]}" == "" ]; then
        /bin/sed -i "s|<JOB_ENV>||g" ${slurm_script}
    else
        /bin/sed -i "s|<JOB_ENV>|export ${MPI_CUDA_JOB_ENV[*]}|g" ${slurm_script}
    fi

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
        echo "Unittests successful (CUDA) MPI (see $testfile for detailed log)"
    fi
fi

# end timer and report time taken
T=$(($SECONDS - $START_TIME))
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0
