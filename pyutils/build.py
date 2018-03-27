#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import perftest


def build_and_test(args):
    import perftest.cmake
    import perftest.config
    import perftest.runtools

    cmake_args = dict()

    # build type
    cmake_args['CMAKE_BUILD_TYPE'] = args.build_type

    # backend selection
    cmake_args['ENABLE_CUDA'] = False
    cmake_args['ENABLE_HOST'] = False
    cmake_args['ENABLE_' + args.backend.upper()] = True

    # precision
    cmake_args['SINGLE_PRECISION'] = args.precision == 'float'

    # grid type
    cmake_args['STRUCTURED_GRIDS'] = args.grid == 'structured'

    # MPI
    cmake_args['USE_MPI'] = args.mpi

    # enable pyutils
    cmake_args['ENABLE_PYUTILS'] = True

    # override cmake args
    for arg in args.cmake:
        k, v = arg.split('=')
        cmake_args[k] = v

    targets = args.targets
    if args.perftest_targets:
        import perftest.stencils
        stencils = perftest.stencils.load(args.grid)
        targets = [s.gridtools_target(args.backend) for s in stencils]

    config = perftest.config.get(args.config)

    # build
    perftest.cmake.build(args.build_dir, args.source_dir, cmake_args,
                         targets, config)

    if not args.run_tests:
        return

    # test
    def command(script):
        path = os.path.join(os.path.abspath(args.build_dir), script)
        return 'sh ' + path

    perftest.runtools.run(command('run_tests.sh'))

    if args.mpi:
        if args.backend == 'cuda':
            perftest.runtools.run(command('run_cuda_mpi_tests.sh'))
        else:
            perftest.runtools.run(command('run_mpi_tests.sh'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='increase verbosity (use -vvv for debug mesages)')

    group = parser.add_argument_group('configuration')
    group.add_argument('--build-type', '-t', choices=['release', 'debug'],
                       required=True)
    group.add_argument('--backend', '-b', choices=['cuda', 'host'],
                       required=True)
    group.add_argument('--precision', '-p', choices=['float', 'double'],
                       required=True)
    group.add_argument('--grid', '-g', choices=['structured', 'icosahedral'],
                       required=True)
    group.add_argument('--cmake', action='append', default=[],
                       metavar='CMAKE_VAR=VALUE')
    group.add_argument('--mpi', action='store_true')
    group.add_argument('--config', '-c')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir_default = os.path.abspath(os.path.join(script_dir,
                                                      os.path.pardir))
    group.add_argument('--source-dir', default=source_dir_default)

    group.add_argument('--build-dir', required=True)

    group = parser.add_argument_group('compilation and testing')

    group = group.add_mutually_exclusive_group()
    group.add_argument('--targets', nargs='+')
    group.add_argument('--perftest-targets', action='store_true')
    group.add_argument('--run-tests', action='store_true')

    args = parser.parse_args()
    perftest.set_verbose(args.verbose)

    with perftest.exception_logging():
        build_and_test(args)
