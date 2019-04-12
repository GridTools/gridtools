#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from pyutils import buildinfo, env, log
import test


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help='increase verbosity (use -vvv for debug mesages)')

parser.add_argument('--run-mpi-tests', '-m', action='store_true')
parser.add_argument('--verbose-ctest', action='store_true')

build_dir_default = os.path.join(buildinfo.binary_dir, 'examples_build')

parser.add_argument('--examples-build-dir', default=build_dir_default)
parser.add_argument('--build-examples', '-b', action='store_true')

args = parser.parse_args()
log.set_verbosity(args.verbose)

with log.exception_logging():
    env.load(buildinfo.target, buildinfo.compiler_id)
    test.run(args.run_mpi_tests, args.verbose_ctest)
    if args.build_examples:
        test.compile_examples(args.examples_build_dir)
