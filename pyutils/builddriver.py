#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import build
import envs
import pyutils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='increase verbosity (use -vvv for debug mesages)')

    parser.add_argument('--build-type', '-b', choices=['release', 'debug'],
                       required=True)
    parser.add_argument('--precision', '-p', choices=['float', 'double'],
                       required=True)
    parser.add_argument('--grid', '-g', choices=['structured', 'icosahedral'],
                       required=True)
    parser.add_argument('--environment', '-e', action='append')
    parser.add_argument('--target', '-t', action='append')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir_default = os.path.abspath(os.path.join(script_dir,
                                                      os.path.pardir))
    parser.add_argument('--source-dir', default=source_dir_default)

    parser.add_argument('--build-dir', '-o', required=True)
    parser.add_argument('--cmake-only', action='store_true')

    args = parser.parse_args()
    pyutils.set_verbose(args.verbose)

    with pyutils.exception_logging():
        env = envs.Env()
        for e in args.environment:
            env.update_from_file(e)
        build.cmake(env, args.source_dir, args.build_dir, args.build_type,
                    args.precision, args.grid)
        if not args.cmake_only:
            build.make(env, args.build_dir, args.target)
