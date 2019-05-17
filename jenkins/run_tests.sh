#!/bin/bash

source $(dirname "$0")/setup.sh

./pyutils/driver.py -v -l $logfile build -b $build_type -p $real_type -g $grid_type -e $envfile -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

./build/pyutils/driver.py -v -l $logfile test $mpi_flag -b || { echo 'Tests failed'; rm -rf $tmpdir; exit 2; }
