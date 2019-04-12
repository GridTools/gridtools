# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import time

from pyutils import log


def cmake(env, source_dir, build_dir, install_dir=None):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f'Source directory "{source_dir}" not found')

    source_dir = os.path.abspath(source_dir)

    build_dir = os.path.abspath(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    if install_dir is not None:
        install_dir = os.path.abspath(install_dir)
        os.makedirs(install_dir, exist_ok=True)

    command = ['cmake', source_dir] + env.cmake_args()
    if install_dir is not None:
        command.append(f'-DCMAKE_INSTALL_PREFIX:STRING={install_dir}')

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


def make(env, build_dir, targets=None, build_command=None):
    if build_command is None:
        build_command = 'make'

    command = build_command.split()
    if targets is not None:
        command += list(targets)

    log.info('Invoking make', ' '.join(command))
    start = time.time()
    try:
        output = subprocess.check_output(command,
                                         env=env,
                                         cwd=build_dir,
                                         stderr=subprocess.STDOUT).decode()
        log.debug('make output', output)
    except subprocess.CalledProcessError as e:
        log.error('make failed with output', e.output.decode())
        raise e
    end = time.time()
    log.info(f'make finished in {end - start:.2f}s')
