#!/bin/bash

#PYTHON_SITE_PACKAGES=$1
#CMAKE_SOURCE_DIR=$2
CMAKE_SOURCE_DIR=$1
#PYTHON_EXECUTABLE=$3
#PYTHON_INSTALL_PREFIX=$4
#echo "building python bindings";
#export PYTHONPATH=${PYTHON_SITE_PACKAGES};
#export GRIDTOOLS_ROOT=${CMAKE_SOURCE_DIR};
#echo "showing Python version" 
#echo "python executable: " $PYTHON_EXECUTABLE
#${PYTHON_EXECUTABLE} --version | cut -d' ' -f2 | cut -d'.' -f1
VERSION=`python --version | cut -d' ' -f2 | cut -d'.' -f1`
echo  $VERSION > ${CMAKE_SOURCE_DIR}/.python_major_version
VERSION=`python --version | cut -d' ' -f2 `
echo  $VERSION > ${CMAKE_SOURCE_DIR}/.python_version
#export PYTHON_VERSION_MAJOR=$VERSION
#cd ${CMAKE_SOURCE_DIR}/python; ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/setup.py build;
#${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/setup.py install --prefix="${PYTHON_INSTALL_PREFIX}";
