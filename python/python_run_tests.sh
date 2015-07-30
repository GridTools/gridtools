#!/bin/bash

#
# remove any files left from the previous run
#
rm -rf /tmp/__gridtools_* > /dev/null 2>&1

CMAKE_SOURCE_DIR=$1
PYTHON_INSTALL_PREFIX=$2

#
# run interactively without arguments
#
if [ -n "${CMAKE_SOURCE_DIR}" ] && [ -n "${PYTHON_INSTALL_PREFIX}" ]; then
    # Checking gcc version (it has to be >=4.8.x)
    gcc_vers=`gcc -dumpversion|cut -f1 -d.`
    if [ $gcc_vers -lt 4 ]
    then
      echo "gcc version 4.x is required. EXIT NOW"
      exit 1
    else
      gcc_vers=`gcc -dumpversion|cut -f2 -d.`
      if [ $gcc_vers -lt 8 ]
      then
        echo "gcc version 4.8.x is required. EXIT NOW"
        exit 1
      else
        # gcc version is 4.8.x
        # Looking for PYTHON_INSTALL_PREFIX
        if [ "$PYTHON_INSTALL_PREFIX" != " " ]
        then
          # Looking for virtualenv
          virtualenv_cmd=`which virtualenv`
          if [ $? -eq 0 ]
          then
            source ${PYTHON_INSTALL_PREFIX}/bin/activate
            export GRIDTOOLS_ROOT=$CMAKE_SOURCE_DIR
          else
            echo "Error while running virtualenv. EXIT NOW"
            exit  1
          fi
        fi 
      fi
    fi
fi

echo "Running Python tests ..."
nosetests -v -s tests.test_sw       & TEST_ONE_PID=$!
nosetests -v -s tests.test_stencils & TEST_TWO_PID=$!
wait "${TEST_ONE_PID}"
TEST_ONE_STATUS=$?
wait "${TEST_TWO_PID}"
TEST_TWO_STATUS=$?
if [ ${TEST_ONE_STATUS} == 0 -a ${TEST_TWO_STATUS} == 0 ]; then
    echo "All Python tests OK"
    if [ -n "${PYTHON_INSTALL_PREFIX}" ]; then
        ${PYTHON_INSTALL_PREFIX}/bin/deactivate
    fi
else
    echo "Error running Python tests. EXIT NOW"
    exit 1
fi
