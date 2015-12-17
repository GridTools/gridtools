#!/bin/bash

CMAKE_SOURCE_DIR=$1
PYTHON_INSTALL_PREFIX=$2


function usage {
    echo "$0 [gridtools source dir] [virtual environment dir]"
    exit 1
}


if [ -z "${CMAKE_SOURCE_DIR}" -o -z "${PYTHON_INSTALL_PREFIX}" ]; then
    usage
else
    #
    # Looking for virtualenv
    #
    virtualenv_cmd=`which virtualenv`
    if [ $? -eq 0 ]; then
        ${virtualenv_cmd} --no-site-packages ${PYTHON_INSTALL_PREFIX}
        source ${PYTHON_INSTALL_PREFIX}/bin/activate
        echo "Installing Python bindings in virtual environment at ${PYTHON_INSTALL_PREFIX} ..."
	#
        # Install python bindings and their dependencies
        #
        which python
        python ${CMAKE_SOURCE_DIR}/python/setup.py install
 	INSTALL_OK=$?
        deactivate
        if [ ${INSTALL_OK} -eq 0 ]; then
           echo "Installation done."
           echo "To use the Python bindings, go to the ${PYTHON_INSTALL_PREFIX} directory and write:"
           echo "$> source bin/activate"
           echo "$> ipython notebook Gridtools4Py"
        else
           echo "Error while installing Python bindings. EXIT NOW"
           exit  1
        fi
    else
        echo "No virtualenv found."
        echo "To install it type: 'pip install virtualenv' and then run cmake again. EXIT NOW"
        exit 1
    fi
fi
