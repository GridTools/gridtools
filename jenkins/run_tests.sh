#!/bin/bash

source $(dirname "$0")/setup.sh

# TODO reset to an appropriate log level (e.g. my making CMake output available already from -v)
./pyutils/driver.py -vvv -l $logfile build -b $build_type -p $real_type -g $grid_type -e $envfile -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

# disable mpi tests by default
if [[ "$run_mpi_tests" == true ]]; then
    run_mpi_tests_flag="--run-mpi-tests"
fi

# build examples by default
if [[ "$build_examples" != false ]]; then
    build_examples_flag="--build-examples"
fi

./build/pyutils/driver.py -v -l $logfile test $run_mpi_tests_flag $build_examples_flag || { echo 'Tests failed'; rm -rf $tmpdir; exit 2; }
