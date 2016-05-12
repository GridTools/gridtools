#!/bin/bash

CMAKE_SOURCE_DIR=$1
PYTHON_INSTALL_PREFIX=$2

#
# remove any files left from the previous run
#
rm -rf /tmp/__gridtools_* > /dev/null 2>&1

#
# run interactively without arguments
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

echo "Running Python tests ..."
#nosetests -v -s tests.test_sw tests.test_stencils
nosetests -v -s tests.test_stencils tests.test_ifstatement tests.test_sw
TEST_STATUS=$?
if [ ${TEST_STATUS} == 0 ]; then
    echo "All Python tests OK"
    if [ -n "${PYTHON_INSTALL_PREFIX}" ]; then
        deactivate
    fi
else
    echo "Error running Python tests. EXIT NOW"
    exit 1
fi
