#!/bin/bash

source $(dirname "$0")/setup.sh

./pyutils/driver.py -v -l $logfile build -b $build_type -p $real_type -g $grid_type -e $envfile -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

if [[ -z "$no_mpi" ]]; then
    mpi_flag="--run-mpi-tests"
fi

if [[ -z "$no_examples" ]]; then
    examples_flag="--build-examples"
fi

./build/pyutils/driver.py -v -l $logfile test $mpi_flag $examples_flag || { echo 'Tests failed'; rm -rf $tmpdir; exit 2; }
