#!/bin/bash
CMAKE_SOURCE_DIR=$1
PYTHON_INSTALL_PREFIX=$2

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
   # Looking for virtualenv
   virtualenv_cmd=`which virtualenv`
   if [ $? -eq 0 ]
   then
       ${virtualenv_cmd} --no-site-packages ${PYTHON_INSTALL_PREFIX}
       source ${PYTHON_INSTALL_PREFIX}/bin/activate
       echo "Installing Python bindings in virtual environment at ${PYTHON_INSTALL_PREFIX} ..."
       which python
       python ${CMAKE_SOURCE_DIR}/python/setup.py install

       if [ $? -eq 0 ]
       then
          deactivate
          echo "Installation done."
          echo "To use the Python bindings, go to the ${PYTHON_INSTALL_PREFIX} directory and write:"
          echo "$> source bin/activate"
          echo "$> ipython notebook Tutorial"
       else
          deactivate
          echo "Error while running python in virtualenv. EXIT NOW"
          exit  1
      fi
   else
     echo "No virtualenv found."
     echo "To install it type: 'pip install virtualenv' and then run cmake again. EXIT NOW"
     exit 1
   fi
  fi
fi
