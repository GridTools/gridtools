# -*- coding: utf-8 -*-

import os
import platform
import re
import subprocess

from pyutils import logger
from pyutils import EnvError, NotFoundError


class Env(dict):
    def __init__(self):
        super().__init__(os.environ.copy())

    def update_from_file(self, envfile):
        if os.path.exists(envfile):
            envdir, envfile = os.path.split(envfile)
        else:
            envdir = os.path.dirname(os.path.abspath(__file__))
            envdir, envfile = os.path.split(os.path.join(envdir, envfile))
            if not os.path.exists(os.path.join(envdir, envfile)):
                raise NotFoundError(f'Expected "{envfile}" at '
                                    f'"{envdir}" not found')

        output = subprocess.check_output(['bash', '-c',
                                          f'source {envfile} && env -0'],
                                          cwd=envdir).decode().strip('\0')
        env = dict(line.split('=', 1) for line in output.split('\0'))
        self.update(env)

        logger.debug(f'Environment updated with {envfile}, new environment:',
                     str(self))

    def __str__(self):
        return '\n'.join(f'{k}={v}' for k, v in sorted(self.items()))

    def _items_with_tag(self, tag):
        return {k[len(tag):]: v for k, v in self.items() if k.startswith(tag)}

    def cmake_args(self):
        return self._items_with_tag('GTCMAKE_')

    def run_settings(self):
        tag = 'GTRUN_'
        return self._items_with_tag(tag)

    @staticmethod
    def hostname():
        """Host name of the current machine.

        Example:
            >>> hostname()
            'keschln-0002'
        """
        hostname = platform.node()
        return hostname

    @staticmethod
    def clustername():
        """SLURM cluster name of the current machine.

        Examples:
            >>> clustername()
            'kesch'
        """
        output = subprocess.check_output(['scontrol', 'show', 'config'])
        p = re.compile(r'.*ClusterName\s*=\s*(\S*).*',
                       re.MULTILINE | re.DOTALL)
        m = p.match(output.decode())
        if not m:
            raise ConfigError('Could not get SLURM cluster name')
        return m.group(1)
