# -*- coding: utf-8 -*-

import os
import subprocess

from pyutils import logger
from pyutils import EnvError, NotFoundError


def load(envfile):
    if os.path.exists(envfile):
        envdir, envfile = os.path.split(envfile)
    else:
        envdir = os.path.dirname(os.path.abspath(__file__))
        envdir, envfile = os.path.split(os.path.join(envdir, envfile))
        if not os.path.exists(os.path.join(envdir, envfile)):
            raise NotFoundError(f'Expected "{envfile}" at "{envdir}" not found')

    output = subprocess.check_output(['bash', '-c',
                                      f'source {envfile} && env -0'],
                                      cwd=envdir).decode().strip('\0')
    env = dict(line.split('=', 1) for line in output.split('\0'))

    logger.debug(f'Environment loaded from {envfile}:',
                 '\n'.join(f'{k}={v}' for k, v in env.items()))
    return env


def _items_with_tag(env, tag):
    return {k[len(tag):]: v for k, v in env.items() if k.startswith(tag)}


def cmake_args(env):
    return _items_with_tag(env, 'GTCMAKE_')


def ci_settings(env):
    tag = 'GTCI_'
    settings =_items_with_tag(env, tag)

    for var in ('QUEUE', 'MPI_NODES', 'MPI_TASKS',
                'BUILD_THREADS', 'BUILD_COMMAND'):
        if var not in settings:
            raise EnvError(f'Missing environment variable {tag}{var}')

    return settings
