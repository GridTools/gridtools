#!/usr/bin/env python

import argparse
import json
import re

import perftest.runtime


def run_tests(runtime, grid, precision, backend, size):
    rt = perftest.runtime.get(runtime, grid, precision, backend)

    result = rt.run(size)

    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--runtime', '-r', choices=['stella', 'gridtools'], required=True)
    parser.add_argument('--backend', '-b', choices=['cpu', 'cuda', 'host'], required=True)
    parser.add_argument('--grid', '-g', choices=['strgrid', 'icgrid'], required=True)
    parser.add_argument('--precision', '-p', choices=['float', 'double'], required=True)
    parser.add_argument('--size', '-s', type=int, nargs=3, required=True)

    args = parser.parse_args()

    run_tests(args.runtime, args.grid, args.precision, args.backend, args.size)
