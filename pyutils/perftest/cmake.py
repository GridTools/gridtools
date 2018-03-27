# -*- coding: utf-8 -*-

import os
import subprocess

from perftest import config, logger, NotFoundError


def configure(build_dir, source_dir, cmake_args, conf=None):
    conf = config.get(conf)

    build_dir = os.path.abspath(build_dir)
    source_dir = os.path.abspath(source_dir)

    if not os.path.isdir(source_dir):
        raise NotFoundError(f'Source directory {source_dir} does not exist')

    os.makedirs(build_dir, exist_ok=True)

    def argstr(k, v):
        if isinstance(v, bool):
            v = 'ON' if v else 'OFF'
        return '-D' + k + '=' + v

    command = conf.cmake_command + [argstr(k, v) for k, v
                                      in cmake_args.items()] + [source_dir]

    logger.debug('Invoking CMake: ' + ' '.join(command))
    try:
        output = subprocess.check_output(command,
                                         env=conf.env,
                                         cwd=build_dir,
                                         stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        logger.error('CMake failed with output:', e.output.decode())
        raise e

    logger.info('CMake finished')
    logger.debug('CMake output:', output)


def make(build_dir, targets=None, conf=None):
    conf = config.get(conf)

    build_dir = os.path.abspath(build_dir)

    command = conf.make_command
    if targets is not None:
        command += targets

    logger.debug('Invoking make: ' + ' '.join(command))
    try:
        output = subprocess.check_output(command, env=conf.env,
                                         cwd=build_dir,
                                         stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        logger.error('CMake failed with output:', e.output.decode())
        raise e

    logger.info('make finished')
    logger.debug('make output:', output)


def build(build_dir, source_dir, cmake_args, targets=None, conf=None):
    conf = config.get(conf)

    configure(build_dir, source_dir, cmake_args, conf)
    make(build_dir, targets, conf)
