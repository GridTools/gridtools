#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

import perftest
from pyutils import envs, log


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


def run(args):
    import perftest.result
    from perftest import buildinfo

    env = envs.Env()
    env.load(buildinfo.target, buildinfo.compiler_id)

    results = perftest.run(env, args.domain_size, args.runs)

    # save result
    if not args.output.lower().endswith('.json'):
        args.output += '.json'

    for tag, result in results.items():
        perftest.result.save(f'.{tag}.'.join(args.output.rsplit('.', 1)),
                             result)


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help='increase verbosity (use -vvv for debug mesages)')

subparsers = parser.add_subparsers(dest='action',
                                   description='action to perform')
subparsers.required = True

# command line aguments for `run` action
run_parser = subparsers.add_parser('run', help='run performance tests')
run_parser.set_defaults(func=run)
run_parser.add_argument('--domain-size', '-s', required=True, type=int,
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
run_parser.add_argument('--device', '-d', choices=['cpu', 'gpu'],
                        required=True)
run_parser.add_argument('--compiler', '-c', choices=['gcc', 'clang', 'icc'],
                        required=True)

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
log.set_verbosity(args.verbose)

with log.exception_logging():
    args.func(args)
