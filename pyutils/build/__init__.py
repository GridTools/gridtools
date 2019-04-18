# -*- coding: utf-8 -*-

import os

from pyutils import env, runtools


def cmake(source_dir, build_dir, install_dir=None):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f'Source directory "{source_dir}" not found')

    source_dir = os.path.abspath(source_dir)

    build_dir = os.path.abspath(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    command = ['cmake', source_dir] + env.cmake_args()
    if install_dir is not None:
        install_dir = os.path.abspath(install_dir)
        os.makedirs(install_dir, exist_ok=True)
        command.append(f'-DCMAKE_INSTALL_PREFIX:STRING={install_dir}')
    runtools.run(command, cwd=build_dir)


def make(build_dir, targets=None):
    command = env.build_command()
    if targets is not None:
        command += list(targets)

    runtools.run(command, cwd=build_dir)
