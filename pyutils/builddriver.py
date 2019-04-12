#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import build
from pyutils import env, log


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help='increase verbosity (use -vvv for debug mesages)')

parser.add_argument('--build-type', '-b', choices=['release', 'debug'],
                   required=True)
parser.add_argument('--precision', '-p', choices=['float', 'double'],
                   required=True)
parser.add_argument('--grid', '-g', choices=['structured', 'icosahedral'],
                   required=True)
parser.add_argument('--device', '-d', choices=['cpu', 'gpu'],
                    required=True)
parser.add_argument('--compiler', '-c', choices=['gnu', 'clang', 'intel'],
                    required=True)
parser.add_argument('--target', '-t', action='append')

script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir_default = os.path.abspath(os.path.join(script_dir,
                                                  os.path.pardir))
parser.add_argument('--source-dir', default=source_dir_default)

parser.add_argument('--build-dir', '-o', required=True)
parser.add_argument('--install-dir', '-i')
parser.add_argument('--cmake-only', action='store_true')

args = parser.parse_args()
log.set_verbosity(args.verbose)

with log.exception_logging():
    env.set_cmake_arg('CMAKE_BUILD_TYPE', args.build_type.title())
    env.set_cmake_arg('GT_SINGLE_PRECISION', args.precision == 'float')
    env.set_cmake_arg('GT_TESTS_ICOSAHEDRAL_GRID', args.grid == 'icosahedral')

    env.load(args.device, args.compiler)

    build.cmake(args.source_dir, args.build_dir, args.install_dir)
    if not args.cmake_only:
        build.make(args.build_dir, args.target)
