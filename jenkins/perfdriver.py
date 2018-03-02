#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import perftest


def plot(args):
    import perftest.plot
    import perftest.result

    results = [perftest.result.load(f) for f in args.input]

    if mode == 'compare':
        fig = perftest.plot.compare(results)
    elif mode == 'history':
        fig = perftest.plot.history(results)

    fig.savefig(args.output)


def run(args):
    import perftest.config
    import perftest.result
    import perftest.runtime

    config = perftest.config.load(args.config)

    rt = config.runtime(args.runtime, args.grid, args.precision, args.backend)
    result = rt.run(args.domain, args.runs)
    perftest.result.save(args.output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0)

    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    subparsers.required = True

    run_parser = subparsers.add_parser('run')
    run_parser.set_defaults(func=run)
    run_parser.add_argument('--runtime', '-r', required=True,
                            choices=['stella', 'gridtools'])
    run_parser.add_argument('--backend', '-b', required=True,
                            choices=['cuda', 'host'])
    run_parser.add_argument('--grid', '-g', required=True,
                            choices=['strgrid', 'icgrid'])
    run_parser.add_argument('--precision', '-p', required=True,
                            choices=['float', 'double'])
    run_parser.add_argument('--domain', '-d', required=True, type=int, nargs=3)
    run_parser.add_argument('--runs', default=10, type=int)
    run_parser.add_argument('--output', '-o', required=True)
    run_parser.add_argument('--config', '-c')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.set_defaults(func=plot)
    plot_parser.add_argument('--mode', '-m', default='compare',
                             choices=['compare', 'history'])
    plot_parser.add_argument('--output', '-o', required=True)
    plot_parser.add_argument('--input', '-i', required=True, nargs='+')

    args = parser.parse_args()
    perftest.set_verbose(args.verbose)

    try:
        args.func(args)
    except Exception:
        perftest.logger.exception('Fatal error: exception was raised')
