# -*- coding: utf-8 -*-

import abc
import json
import os
import re
import subprocess

from perftest import NotFoundError, ParseError, ArgumentError
from perftest import runtools, stencils, utils


class Result:
    def __init__(self, filename=None, **kwargs):
        if filename:
            with open(filename, 'r') as fp:
                data = json.load(fp)
            data.update(kwargs)
        else:
            data = kwargs
        self.fields = frozenset(data.keys())
        for field in self.fields:
            setattr(self, field, data[field])

    def write(self, filename):
        data = {field: getattr(self, field) for field in self.fields}
        with open(filename, 'w') as fp:
            json.dump(data, fp, indent=4)

    def __str__(self):
        fstr = ', '.join(f'{f}={getattr(self, f)}' for f in self.fields)
        return f'Result({fstr})'

    def has_same_stencils(self, *others):
        stimekeys = set(self.times.keys())
        otimekeys = [set(o.times.keys()) for o in others]
        return stimekeys == stimekeys.intersection(*otimekeys)

    def common_fields(self, *others):
        candidates = self.fields.intersection(*(o.fields for o in others))
        common = set()
        for f in candidates:
            if all(getattr(self, f) == getattr(o, f) for o in others):
                common.add(f)
        return common

    def get_runtime(self):
        return get(self.runtime, self.grid, self.precision, self.backend)


class Runtime(metaclass=abc.ABCMeta):
    def __init__(self, grid, precision, backend):
        self.grid = grid
        self.precision = precision
        self.backend = backend

    def run(self, size):
        stencil_list = stencils.instantiate(self.grid)
        commands = [self.command(s, size) for s in stencil_list]
        outputs = runtools.run(commands)
        times = {s.name: self._parse_time(o) for s, o in zip(stencil_list,
                                                             outputs)}
        return Result(runtime=self.name,
                      version=self.version,
                      backend=self.backend,
                      grid=self.grid,
                      precision=self.precision,
                      size=size,
                      times=times,
                      timestamp=utils.get_timestamp())

    def _parse_time(self, output):
        p = re.compile(r'.*\[s\]\s*([0-9.]+).*', re.MULTILINE | re.DOTALL)
        m = p.match(output)
        if not m:
            raise ParseError(f'Could not parse time in output:\n{output}')
        return float(m.group(1))

    def __str__(self):
        return (f'Runtime(name={self.name}, version={self.version}, '
                f'path={self.path})')

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
        binary = os.path.join(self.path, f'StandaloneStencils{suffix}')
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
                                       cwd=self.path).decode().strip()

    def binary(self, stencil):
        binary = getattr(stencil, 'gridtools_' + self.backend.lower())
        binary = os.path.join(self.path, binary)
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
