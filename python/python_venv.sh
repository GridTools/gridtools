#!/bin/bash
#PYTHON_SITE_PACKAGES=$1
#CMAKE_SOURCE_DIR=$2
CMAKE_SOURCE_DIR=$1
#PYTHON_EXECUTABLE=$3
#PYTHON_INSTALL_PREFIX=$4
PYTHON_INSTALL_PREFIX=$2

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
   # Looking for virtualenv
   virtualenv_cmd=`which virtualenv`
   if [ $? -eq 0 ]
   then
     echo $virtualenv_cmd
     $virtualenv_cmd --no-site-packages ${PYTHON_INSTALL_PREFIX}
     if [ $? -ne 0 ]
     then 
       echo "Error while installing virtualenv. EXIT NOW"
       exit 1
     else
       echo "Virtual environment installed in " ${PYTHON_INSTALL_PREFIX}
     fi
   else
     echo "No virtualenv found."
     echo "To install it type: 'pip install virtualenv' and then run cmake again. EXIT NOW"
     exit 1
   fi
  fi
fi
