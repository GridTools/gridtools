#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

import perftest


def plot(args):
    import perftest.plot
    import perftest.result

    # load results from files
    results = [perftest.result.load(f) for f in args.input]

    # plot
    if args.mode == 'compare':
        fig = perftest.plot.compare(results)
    elif args.mode == 'history':
        fig = perftest.plot.history(results, args.date, args.limit)

    # save result
    fig.savefig(args.output)


def validate(args):
    import perftest.result
    import perftest.validate

    res_a, res_b = (perftest.result.load(f) for f in args.input)

    if not perftest.validate.ttest_validate(res_a, res_b, 0.9):
        sys.exit(2)


def run(args):
    import perftest.config
    import perftest.result
    import perftest.runtime

    # load configuration for current machine
    config = perftest.config.get(args.config)

    # get runtime
    rt = config.runtime(args.runtime)

    # run jobs
    result = rt.run(args.domain, args.runs, args.max_parallel_jobs)

    # save result
    if not args.output.lower().endswith('.json'):
        args.output += '.json'
    perftest.result.save(args.output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='increase verbosity (use -vvv for debug mesages)')

    subparsers = parser.add_subparsers(dest='action',
                                       description='action to perform')
    subparsers.required = True

    # command line aguments for `run` action
    run_parser = subparsers.add_parser('run', help='run performance tests')
    run_parser.set_defaults(func=run)
    run_parser.add_argument('--runtime', '-r', default='gridtools',
                            choices=['stella', 'gridtools'],
                            help='runtime system to use')
    run_parser.add_argument('--domain', '-d', required=True, type=int,
                            nargs=3, metavar=('ISIZE', 'JSIZE', 'KSIZE'),
                            help='domain size (excluding halo)')
    run_parser.add_argument('--runs', default=10, type=int,
                            help='number of runs to do for each stencil')
    run_parser.add_argument('--max-parallel-jobs', default=50, type=int,
                            help='max number of jobs that are submitted to '
                                 'SLURM in parallel')
    run_parser.add_argument('--output', '-o', required=True,
                            help='output file path, extension .json is added '
                                 'if not given')
    run_parser.add_argument('--config', '-c',
                            help='config name, default is machine config')

    # command line arguments for `validate` action
    validate_parser = subparsers.add_parser('validate',
                                            help='statistically validate '
                                                 'performance results')
    validate_parser.set_defaults(func=validate)
    validate_parser.add_argument('--input', '-i', required=True, nargs=2,
                                 help='input result files')

    # command line aguments for `plot` action
    plot_parser = subparsers.add_parser('plot',
                                        help='plot performance results')
    plot_parser.set_defaults(func=plot)

    plot_subparsers = plot_parser.add_subparsers(dest='mode',
                                                 description='plotting mode')
    plot_subparsers.required = True

    # command line arguments for `plot compare` action
    plot_compare_parser = plot_subparsers.add_parser('compare',
                                                     help='run time '
                                                          'comparison per '
                                                          'stencil')
    plot_compare_parser.add_argument('--output', '-o', required=True,
                                     help='output file, can have any extension'
                                          ' supported by matplotlib')
    plot_compare_parser.add_argument('--input', '-i', required=True, nargs='+',
                                     help='any number of input files')

    # command line arguments for `plot history` action
    plot_history_parser = plot_subparsers.add_parser('history',
                                                     help='run time history')
    plot_history_parser.add_argument('--output', '-o', required=True,
                                     help='output file, can have any extension'
                                          ' supported by matplotlib')
    plot_history_parser.add_argument('--input', '-i', required=True, nargs='+',
                                     help='any number of input files')
    plot_history_parser.add_argument('--date', '-d', default='job',
                                     choices=['runtime', 'job'],
                                     help='date to use, either the '
                                          'build/commit date of the runtime '
                                          'or the date when the job was run')
    plot_history_parser.add_argument('--limit', '-l', type=int,
                                     help='limit the history size to the '
                                          'given number of results')

    args = parser.parse_args()
    perftest.set_verbose(args.verbose)

    with perftest.exception_logging():
        args.func(args)
