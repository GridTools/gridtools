#!/bin/bash

CMAKE_SOURCE_DIR=$1
VERSION=`python --version | cut -d' ' -f2 | cut -d'.' -f1`
echo  $VERSION > ${CMAKE_SOURCE_DIR}/.python_major_version
VERSION=`python --version | cut -d' ' -f2 `
echo  $VERSION > ${CMAKE_SOURCE_DIR}/.python_version
