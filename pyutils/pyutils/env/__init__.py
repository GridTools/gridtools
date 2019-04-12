# -*- coding: utf-8 -*-

import os
import platform
import re

from pyutils import log, runtools


env = os.environ.copy()


def load(target, compiler):
    envfile = clustername() + '_' + target + '_' + compiler + '.sh'
    envfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           envfile)
    update_from_file(envfile)
    env['GTCMAKE_PYUTILS_TARGET'] = target


def update_from_file(envfile):
    if not os.path.exists(envfile):
        raise FileNotFoundError(f'Could find environment file "{envfile}"')

    envdir, envfile = os.path.split(envfile)
    output = runtools.run(['bash', '-c', f'source {envfile} && env -0'],
                          cwd=envdir).strip('\0')
    env.update(line.split('=', 1) for line in output.split('\0'))

    log.debug(f'Environment updated with {envfile}, new environment',
              '\n'.join(f'{k}={v}' for k, v in sorted(env.items())))


def _items_with_tag(tag):
    return {k[len(tag):]: v for k, v in env.items() if k.startswith(tag)}


def cmake_args():
    args = []
    for k, v in _items_with_tag('GTCMAKE_').items():
        if v.strip() in ('ON', 'OFF'):
            k += ':BOOL'
        else:
            k += ':STRING'
        args.append(f'-D{k}={v}')
    return args


def set_cmake_arg(arg, value):
    if isinstance(value, bool):
        value = 'ON' if value else 'OFF'
    env['GTCMAKE_' + arg] = value


def sbatch_options(mpi):
    options = _items_with_tag('GTRUN_SBATCH_')
    if mpi:
        options.update(_items_with_tag('GTRUNMPI_SBATCH_'))

    return ['--' + k.lower().replace('_', '-') + '=' + v
            for k, v in options.items()]


def srun_command():
    return (env.get('GTCMAKE_MPITEST_EXECUTABLE', 'srun') + ' '
            + env.get('GTCMAKE_MPITEST_PREFLAGS', '').replace(';', ' '))


def build_command():
    return env.get('GTRUN_BUILD_COMMAND', 'make').split()


def hostname():
    """Host name of the current machine.

    Example:
        >>> hostname()
        'keschln-0002'
    """
    hostname = platform.node()
    return hostname


def clustername():
    """SLURM cluster name of the current machine.

    Examples:
        >>> clustername()
        'kesch'
    """
    output = runtools.run(['scontrol', 'show', 'config'])
    m = re.compile(r'.*ClusterName\s*=\s*(\S*).*',
                   re.MULTILINE | re.DOTALL).match(output)
    if not m:
        raise EnvironmentError('Could not get SLURM cluster name')
    return m.group(1)
