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

    def load(self, target, compiler):
        envfile = self.clustername() + '_' + target + '_' + compiler + '.sh'
        envfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               envfile)
        self.update_from_file(envfile)
        self['GTCMAKE_PYUTILS_TARGET'] = target

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
        args = []
        for k, v in self._items_with_tag('GTCMAKE_').items():
            if v.strip() in ('ON', 'OFF'):
                k += ':BOOL'
            else:
                k += ':STRING'
            args.append(f'-D{k}={v}')
        return args

    def sbatch_options(self, mpi):
        options = []
        def fmt(k, v):
            return '--' + k.lower().replace('_', '-') + '=' + v
        for k, v in self._items_with_tag('GTRUN_SBATCH_').items():
            options.append(fmt(k, v))
        if mpi:
            for k, v in self._items_with_tag('GTRUNMPI_SBATCH_').items():
                options.append(fmt(k, v))
        return options

    def srun_command(self):
        return (self.get('GTCMAKE_MPITEST_EXECUTABLE', 'srun') + ' '
                + self.get('GTCMAKE_MPITEST_PREFLAGS', ''))

    def build_command(self):
        return self.get('GTRUN_BUILD_COMMAND', 'make')

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
