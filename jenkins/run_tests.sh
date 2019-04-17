#!/bin/bash

source $(dirname "$0")/setup.sh

./pyutils/builddriver.py -v -l $logfile -b $build_type -p $real_type -g $grid_type -e $envfile -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

./build/pyutils/testdriver.py -v -l $logfile -m -b || { echo 'Tests failed'; rm -rf $tmpdir; exit 2; }
