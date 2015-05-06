#!/bin/bash

#
# remove any files left from the previous run
#
rm -rf /tmp/__gridtools_* 

CMAKE_SOURCE_DIR=$1
PYTHON_INSTALL_PREFIX=$2

#
# run interactively without arguments
#
if [ -n "${CMAKE_SOURCE_DIR}" ] && [ -n "${PYTHON_INSTALL_PREFIX}" ]; then
    # Checking gcc version (it has to be >=4.9.x)
    gcc_vers=`gcc -dumpversion|cut -f1 -d.`
    if [ $gcc_vers -lt 4 ]
    then
      echo "gcc version 4.x is required. EXIT NOW"
      exit 1
    else
      gcc_vers=`gcc -dumpversion|cut -f2 -d.`
      if [ $gcc_vers -lt 9 ]
      then
        echo "gcc version 4.9.x is required. EXIT NOW"
        exit 1
      else
        # gcc version is 4.9.x
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

echo "Running python tests ..."
nosetests -v -s -x --with-coverage --cover-package=gridtools --cover-erase --cover-html tests.test_combined_stencils tests.test_stencils tests.test_sw
if [ $? -ne 0 ]; then
    echo "Error running python tests. EXIT NOW"
    exit 1
else
    if [ -n "${PYTHON_INSTALL_PREFIX}" ]; then
        ${PYTHON_INSTALL_PREFIX}/bin/deactivate
    fi
fi
