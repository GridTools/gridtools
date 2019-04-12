#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from pyutils import buildinfo, envs, log
import test


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help='increase verbosity (use -vvv for debug mesages)')

parser.add_argument('--run-mpi-tests', '-m', action='store_true')
parser.add_argument('--verbose-ctest', action='store_true')

script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir_default = os.path.abspath(os.path.join(script_dir, os.path.pardir))

parser.add_argument('--build-dir', default=build_dir_default)

args = parser.parse_args()
log.set_verbosity(args.verbose)

with log.exception_logging():
    env = envs.Env()
    env.load(buildinfo.target, buildinfo.compiler_id)
    test.run(env, args.run_mpi_tests, args.verbose_ctest)
