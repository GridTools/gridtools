#!/bin/bash

source $(dirname "$0")/setup.sh

./pyutils/builddriver.py -vvv -b $build_type -p $real_type -g $grid_type -d $target -c $compiler -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

./build/pyutils/testdriver.py -vvv -m -b || { echo 'Tests failed'; rm -rf $tmpdir; exit 2; }
