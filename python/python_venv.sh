#!/bin/bash

PYTHON_INSTALL_PREFIX=$1

if [ -n "${PYTHON_INSTALL_PREFIX}" ]; then
    #
    # Looking for virtualenv
    #
    virtualenv_cmd=`which virtualenv`
    if [ $? -eq 0 ]; then
         echo $virtualenv_cmd
         $virtualenv_cmd --no-site-packages ${PYTHON_INSTALL_PREFIX}
         if [ $? -ne 0 ]; then
           echo "Error while creating virtualenv. EXIT NOW"
           exit 1
         else
           echo "Virtual environment installed in " ${PYTHON_INSTALL_PREFIX}
         fi
    else
         echo "No virtualenv found."
         echo "To install it type: 'pip install virtualenv' and then run cmake again. EXIT NOW"
         exit 1
    fi
else
    echo "$0 [directory]"
    exit 1
fi
