#!/usr/bin/env python

import argparse
import re

import perftest.config.machine as machine
import perftest.config.stencils as stencils
import perftest.runtools as runtools


def parse_time(output):
    p = re.compile(r'.*\[s\]\s*([0-9.]+).*', re.MULTILINE | re.DOTALL)
    m = p.match(output)
    if not m:
        raise ValueError(f'Could not parse time in output:\n{output}')
    return float(m.group(1))


def run_tests(runtime, backend, grid_type, precision, size):
    stencil_list = stencils.instantiate(grid_type)

    commands = [machine.command(grid_type, precision, runtime, backend, stencil, size)
                for stencil in stencil_list]

    times = [parse_time(output) for output in runtools.run_multiple(commands)]

    for stencil, time in zip(stencil_list, times):
        print(f'Result for {stencil.name}: {time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--runtime', '-r', choices=['stella', 'gridtools'], required=True)
    parser.add_argument('--backend', '-b', choices=['cpu', 'cuda', 'host'], required=True)
    parser.add_argument('--grid-type', '-g', choices=['strgrid', 'icgrid'], required=True)
    parser.add_argument('--precision', '-p', choices=['float', 'double'], required=True)
    parser.add_argument('--size', '-s', type=int, nargs=3, required=True)

    args = parser.parse_args()

    run_tests(args.runtime, args.backend, args.grid_type, args.precision, args.size)
