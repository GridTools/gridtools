#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import perftest.runtime
import perftest.plot


def plot_compare(infiles, outfile):
    results = [perftest.runtime.Result(f) for f in infiles]
    fig = perftest.plot.compare(*results)
    fig.savefig(outfile)


def run(runtime, grid, precision, backend, size, outfile):
    rt = perftest.runtime.get(runtime, grid, precision, backend)
    rt.run(size).write(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--runtime', '-r', required=True,
                            choices=['stella', 'gridtools'])
    run_parser.add_argument('--backend', '-b', required=True,
                            choices=['cpu', 'cuda', 'host'])
    run_parser.add_argument('--grid', '-g', required=True,
                            choices=['strgrid', 'icgrid'])
    run_parser.add_argument('--precision', '-p', required=True,
                            choices=['float', 'double'])
    run_parser.add_argument('--size', '-s', required=True, type=int, nargs=3)
    run_parser.add_argument('--output', '-o', required=True)

    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument('--mode', '-m', default='compare',
                             choices=['compare'])
    plot_parser.add_argument('--output', '-o', required=True)
    plot_parser.add_argument('--input', '-i', required=True, nargs='+')

    args = parser.parse_args()

    if args.command == 'run':
        run(args.runtime, args.grid, args.precision,
            args.backend, args.size, args.output)
    elif args.command == 'plot':
        if args.mode == 'compare':
            plot_compare(args.input, args.output)
