# -*- coding: utf-8 -*-

import argparse
import os
import subprocess

from pyutils import logger
import envs


def configure(env, source_dir, build_dir, build_type, precision, grid_type):
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

    args = envs.cmake_args(env)
    command = ['cmake', source_dir] + [f'-D{k}={v}' for k, v in args.items()]

    logger.info('Invoking CMake: ' + ' '.join(command))
    try:
        output = subprocess.check_output(command,
                                         env=env,
                                         cwd=build_dir,
                                         stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        logger.error('CMake failed with output:', e.output.decode())
        raise e

    logger.info('CMake finished')
    logger.debug('CMake output:', output)


def make(env, build_dir, targets=None):
    ci_settings = envs.ci_settings(env)
    threads = ci_settings['BUILD_THREADS']
    command = ci_settings['BUILD_COMMAND']

    command = list(command.split()) + ['make', f'-j{threads}']
    if targets is not None:
        command += list(targets)

    logger.info('Invoking make : ' + ' '.join(command))
    try:
        output = subprocess.check_output(command,
                                         env=env,
                                         cwd=build_dir,
                                         stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        logger.error('make failed with output:', e.output.decode())

    logger.info('make finished')
    logger.debug('make output:', output)


def build(env, source_dir, build_dir, build_type, precision, grid_type,
          targets=None):
    configure(env, source_dir, build_dir, build_type, precision, grid_type)
    make(env, build_dir, targets)
