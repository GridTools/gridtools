# -*- coding: utf-8 -*-

import abc
import os
import re
import statistics
import subprocess

from perftest import NotFoundError, ParseError, ArgumentError
from perftest import logger, result, runtools, stencils, utils


class Runtime(metaclass=abc.ABCMeta):
    def __init__(self, grid, precision, backend):
        if grid not in self.grids:
            sup = ', '.join(f'"{g}"' for g in self.grids)
            raise ArgumentError(
                    f'Invalid grid "{grid}", supported are {sup}')
        if precision not in self.precisions:
            sup = ', '.join(f'"{p}"' for p in self.precisions)
            raise ArgumentError(
                    f'Invalid precision "{precision}", supported are {sup}')
        if backend not in self.backends:
            sup = ', '.join(f'"{b}"' for b in self.backends)
            raise ArgumentError(
                    f'Invalid backend "{backend}", supported are {sup}')
        self.grid = grid
        self.precision = precision
        self.backend = backend

    def run(self, domain, runs):
        stencil_list = stencils.instantiate(self.grid)
        commands = [self.command(s, domain) for s in stencil_list]

        allcommands = [c for c in commands for _ in range(runs)]
        logger.info('Running stencils')
        alloutputs = runtools.run(allcommands)
        logger.info('Running stencils finished')

        alltimes = [self._parse_time(o) for o in alloutputs]

        times = [alltimes[i:i+runs] for i in range(0, len(alltimes), runs)]

        meantimes = [statistics.mean(t) for t in times]
        stdevtimes = [statistics.stdev(t) if len(t) > 1 else 0 for t in times]

        return result.Result(runtime=self,
                             domain=domain,
                             stencils=stencil_list,
                             meantimes=meantimes,
                             stdevtimes=stdevtimes)

    @staticmethod
    def _parse_time(output):
        p = re.compile(r'.*\[s\]\s*([0-9.]+).*', re.MULTILINE | re.DOTALL)
        m = p.match(output)
        if not m:
            raise ParseError(f'Could not parse time in output:\n{output}')
        return float(m.group(1))

    def __str__(self):
        return (f'Runtime(name={self.name}, version={self.version}, '
                f'datetime={self.datetime}, path={self.path})')

    @property
    def name(self):
        return re.sub(r'(.*)Runtime', r'\1', self.__class__.__name__).lower()

    @abc.abstractproperty
    def version(self):
        pass

    @abc.abstractproperty
    def datetime(self):
        pass

    @abc.abstractmethod
    def path(self):
        pass

    @abc.abstractmethod
    def binary(self, stencil):
        pass

    @abc.abstractmethod
    def command(self, stencil, domain):
        pass


class StellaRuntimeBase(Runtime):
    grids = ['strgrid']
    precisions = ['float', 'double']
    backends = ['cuda', 'host']

    def binary(self, stencil):
        suffix = 'CUDA' if self.backend == 'cuda' else ''
        binary = os.path.join(self.path, f'StandaloneStencils{suffix}')
        if not os.path.isfile(binary):
            raise NotFoundError(f'Could not find STELLA binary at {binary}')
        return binary

    def command(self, stencil, domain):
        binary = self.binary(stencil)
        ni, nj, nk = domain
        filt = stencil.stella_filter
        return f'{binary} --ie {ni} --je {nj} --ke {nk} --gtest_filter={filt}'


class GridtoolsRuntimeBase(Runtime):
    grids = ['strgrid', 'icgrid']
    precisions = ['float', 'double']
    backends = ['cuda', 'host']

    @property
    def version(self):
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=self.path).decode().strip()

    @property
    def datetime(self):
        commit = self.version
        posixtime = subprocess.check_output(['git', 'show', '-s',
                                             '--format=%ct', commit],
                                            cwd=self.path).decode().strip()
        return utils.timestr_from_posix(posixtime)

    def binary(self, stencil):
        binary = getattr(stencil, 'gridtools_' + self.backend.lower())
        binary = os.path.join(self.path, binary)
        if not os.path.isfile(binary):
            raise NotFoundError(f'Could not find GridTools binary at {binary}')
        return binary

    def command(self, stencil, domain):
        binary = self.binary(stencil)
        ni, nj, nk = domain
        halo = stencil.halo
        ni, nj, nk = ni + 2 * halo, nj + 2 * halo, nk + 2 * halo
        return f'{binary} {ni} {nj} {nk} 10'


def get(runtime, grid, precision, backend):
    from perftest import config
    cls = config.get_runtime(runtime)
    return cls(grid=grid, precision=precision, backend=backend)
