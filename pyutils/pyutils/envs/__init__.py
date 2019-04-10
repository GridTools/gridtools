# -*- coding: utf-8 -*-

import os
import platform
import re
import subprocess

from pyutils import log


class Env(dict):
    def __init__(self, other=None):
        if other is not None:
            super().__init__(other)
        else:
            super().__init__(os.environ.copy())

    def copy(self):
        return Env(self)

    def load(self, device, compiler):
        envfile = self.clustername() + '_' + device + '_' + compiler + '.sh'
        envfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               envfile)
        self.update_from_file(envfile)

    def update_from_file(self, envfile):
        if not os.path.exists(envfile):
            raise FileNotFoundError(f'Could find environment file "{envfile}"')

        envdir, envfile = os.path.split(envfile)
        output = subprocess.check_output(['bash', '-c',
                                          f'source {envfile} && env -0'],
                                          cwd=envdir).decode().strip('\0')
        env = dict(line.split('=', 1) for line in output.split('\0'))
        self.update(env)

        log.debug(f'Environment updated with {envfile}, new environment',
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
            raise EnvironmentError('Could not get SLURM cluster name')
        return m.group(1)
