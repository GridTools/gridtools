#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from pyutils import envs, log
import test


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help='increase verbosity (use -vvv for debug mesages)')

parser.add_argument('--device', '-d', choices=['cpu', 'gpu'],
                    required=True)
parser.add_argument('--compiler', '-c', choices=['gcc', 'clang', 'icc'],
                    required=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir_default = os.path.abspath(os.path.join(script_dir, os.path.pardir))

parser.add_argument('--build-dir', default=build_dir_default)

args = parser.parse_args()
log.set_verbosity(args.verbose)

with log.exception_logging():
    env = envs.Env()
    env.load(args.device, args.compiler)
    test.run(env):
