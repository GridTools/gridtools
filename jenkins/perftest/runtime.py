# -*- coding: utf-8 -*-

import abc
import os
import re
import subprocess

from perftest import NotFoundError, ParseError
from perftest import runtools, stencils


class Runtime(metaclass=abc.ABCMeta):
    def __init__(self, grid, precision, backend):
        self.grid = grid
        self.precision = precision
        self.backend = backend

    def run(self, size):
        all_stencils = stencils.instantiate(self.grid)
        commands = [self.command(s, size) for s in all_stencils]
        outputs = runtools.run(commands)
        times = {s.name: self._parse_time(o) for s, o in zip(all_stencils, outputs)}
        return {'runtime': self.name,
                'version': self.version,
                'backend': self.backend,
                'grid': self.grid,
                'precision': self.precision,
                'size': size,

    def _parse_time(self, output):
        p = re.compile(r'.*\[s\]\s*([0-9.]+).*', re.MULTILINE | re.DOTALL)
        m = p.match(output)
        if not m:
            raise ParseError(f'Could not parse time in output:\n{output}')
        return float(m.group(1))
                'times': times}

    @property
    def name(self):
        return re.sub(r'(.*)Runtime', r'\1', self.__class__.__name__).lower()

    @abc.abstractproperty
    def version(self):
        pass

    @abc.abstractmethod
    def path(self):
        pass

    @abc.abstractmethod
    def binary(self, stencil):
        pass

    @abc.abstractmethod
    def command(self, stencil, size):
        pass


class StellaRuntimeBase(Runtime):
    def binary(self, stencil):
        suffix = 'CUDA' if self.backend == 'cuda' else ''
        binary = os.path.join(self.path(), f'StandaloneStencils{suffix}')
        if not os.path.isfile(binary):
            raise NotFoundError(f'Could not find STELLA binary at {binary}')
        return binary

    def command(self, stencil, size):
        binary = self.binary(stencil)
        ni, nj, nk = size
        filt = stencil.stella_filter
        return f'{binary} --ie {ni} --je {nj} --ke {nk} --gtest_filter={filt}'


class GridtoolsRuntimeBase(Runtime):
    @property
    def version(self):
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=self.path()).decode().strip()

    def binary(self, stencil):
        binary = getattr(stencil, 'gridtools_' + self.backend.lower())
        binary = os.path.join(self.path(), binary)
        if not os.path.isfile(binary):
            raise NotFoundError(f'Could not find GridTools binary at {binary}')
        return binary

    def command(self, stencil, size):
        binary = self.binary(stencil)
        ni, nj, nk = size
        halo = stencil.halo
        ni, nj, nk = ni + 2 * halo, nj + 2 * halo, nk + 2 * halo
        return f'{binary} {ni} {nj} {nk} 10'


from perftest import machine


def get(runtime, grid, precision, backend):
    try:
        cls = getattr(machine, runtime.title() + 'Runtime')
    except AttributeError:
        raise ArgumentError(f'Runtime "{runtime}" not available')
    return cls(grid, precision, backend)

