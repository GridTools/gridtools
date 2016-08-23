#!/bin/bash

DISABLE_GPU=false

#
# Parse the command line arguments looking for recognized options
#
for i in "$@"; do
  case $i in
    --no-gpu)   #Do not run CUDA tests
    DISABLE_GPU=true
    shift
    ;;
    *)          #Leave all other arguments as they are

    ;;
  esac
done

#
# After parsing the options, the shifts should have left just 2 optional arguments
# that can be parsed positionally
#
CMAKE_SOURCE_DIR=$1
PYTHON_INSTALL_PREFIX=$2

#
# remove files left from the previous runs which are older than two days
#
find /tmp -iname '__gridtools_*' -type d -ctime +2 -exec rm -rf {} \; > /dev/null 2>&1
find /tmp -iname 'gt4py_test-*.log' -type f -ctime +2 -exec rm -rf {} \; > /dev/null 2>&1

#
# run interactively if no arguments given
#
if [ -n "${CMAKE_SOURCE_DIR}" ] && [ -n "${PYTHON_INSTALL_PREFIX}" ]; then
    # Looking for PYTHON_INSTALL_PREFIX
    if [ "$PYTHON_INSTALL_PREFIX" != " " ]
    then
      # Looking for virtualenv
      virtualenv_cmd=`which virtualenv`
      if [ $? -eq 0 ]
      then
        source ${PYTHON_INSTALL_PREFIX}/bin/activate
        echo "Activated virtual environment ($VIRTUAL_ENV)"
      else
        echo "Error while activating virtualenv. EXIT NOW"
        exit  1
      fi
    fi
fi

#
# Exclude GPU tests, if requested, by creating a string with the appropriate
# condition to be added as an option when launching Nose
#
NOSE_NO_GPU=""
if [ "$DISABLE_GPU" == true ] ; then
  NOSE_NO_GPU="-A 'lang != \"cuda\"'"
  echo "Excluding GPU tests"
fi

# Create commands as strings to be executed using "eval" to avoid some
# obscure string conversion mechanisms that cause them to fail otherwise
NOSE_CMD[1]="nosetests -v -s ${NOSE_NO_GPU} tests.test_stencils"
NOSE_CMD[2]="nosetests -v -s ${NOSE_NO_GPU} tests.test_ifstatement tests.test_fastwaves"
NOSE_CMD[3]="nosetests -v -s ${NOSE_NO_GPU} tests.test_sw"
NOSE_CMD[4]="nosetests -v -s ${NOSE_NO_GPU} tests.test_gtdd tests.test_invalid_stencils"

#
# Run the tests using multiple background processes
# The PID of each process is collected to check the exit status and to
# build a string with the kill commands for all processes
#
START_TIME=$(date +%s)
KILL_STRING=""
echo "Running Python tests ..."
for i in `seq 1 ${#NOSE_CMD[@]}`; do
    eval "${NOSE_CMD[$i]} > /tmp/gt4py_test.log 2>&1 &"
    mv /tmp/gt4py_test.log /tmp/gt4py_test-$!.log
    PIDS[$i]=$!
    KILL_STRING="$KILL_STRING kill ${PIDS[$i]};"
done

#
# Connect the execution of the KILL_STRING with the reception
# of the Interrupt Signal (SIGINT), sent with Ctrl-C.
#
trap "${KILL_STRING}" SIGINT

#
# Wait for processes to finish and collect their exit statuses
#
for i in `seq 1 ${#PIDS[@]}`; do
    wait "${PIDS[$i]}"
    TEST_STATUS[$i]=$?
done

#
# Display running time
#
END_TIME=$(date +%s)
echo "Ran tests in $(($END_TIME - $START_TIME)) seconds"

#
# Clear commands bound to the Interrupt Signal
#
trap - SIGINT

#
# Check tests exit statuses
#
ALL_OK=0
for i in `seq 1 ${#PIDS[@]}`; do
    if [ "${TEST_STATUS[$i]}" != 0 ]; then
        echo "Errors detected in runnning ${NOSE_CMD[$i]}. See file gt4py_test-${PIDS[$i]}.log for details."
        ALL_OK=1
    fi
done

#
# Exit from script
#
if [ ${ALL_OK} == 0 ]; then
    echo "All Python tests OK"
    if [ -n "${PYTHON_INSTALL_PREFIX}" ]; then
        deactivate
    fi
else
    echo "Error running Python tests. EXIT NOW"
    exit 1
fi
