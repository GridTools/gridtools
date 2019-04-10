# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import time

from pyutils import log


def cmake(env, source_dir, build_dir, build_type, precision, grid_type):
    if build_type not in ('debug', 'release'):
        raise ValueError(f'Invalid build type "{build_type}"')
    if precision not in ('float', 'double'):
        raise ValueError(f'Invalid precision "{precision}"')
    if grid_type not in ('structured', 'icosahedral'):
        raise ValueError(f'Invalid grid type "{grid_type}"')

    source_dir = os.path.abspath(source_dir)
    build_dir = os.path.abspath(build_dir)

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f'Source directory "{source_dir}" not found')

    os.makedirs(build_dir, exist_ok=True)

    args = env.cmake_args()
    command = ['cmake', source_dir] + [f'-D{k}={v}' for k, v in args.items()]

    log.info('Invoking CMake', ' '.join(command))
    start = time.time()
    try:
        output = subprocess.check_output(command,
                                         env=env,
                                         cwd=build_dir,
                                         stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        log.error('CMake failed with output', e.output.decode())
        raise e
    end = time.time()
    log.info(f'CMake finished in {end - start:.2f}s')
    log.debug('CMake output', output)


def make(env, build_dir, targets=None):
    run_settings = env.run_settings()
    threads = run_settings['BUILD_THREADS']
    command = run_settings['BUILD_COMMAND']

    command = list(command.split()) + ['make', f'-j{threads}']
    if targets is not None:
        command += list(targets)

    log.info('Invoking make', ' '.join(command))
    start = time.time()
    try:
        output = subprocess.check_output(command,
                                         env=env,
                                         cwd=build_dir,
                                         stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        log.error('make failed with output', e.output.decode())
    end = time.time()
    log.info(f'make finished in {end - start:.2f}s')
    log.debug('make output', output)
