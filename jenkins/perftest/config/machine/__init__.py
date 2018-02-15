# -*- coding: utf-8 -*-

import importlib
import os
import platform
import re


def name():
    hostname = platform.node()
    return re.sub(r'^([a-z]+)(ln-)?\d*$', '\g<1>', hostname)


system = importlib.import_module('perftest.config.machine.' + name())


sbatch = system.sbatch


def stella_path(grid, precision, backend):
    assert grid == 'strgrid'
    assert precision in {'float', 'double'}
    assert backend in {'cpu', 'cuda'}
    return system.stella_path(grid, precision, backend)


def stella_binary(grid, precision, backend):
    if backend == 'cuda':
        suffix = backend.upper()
    else:
        suffix = ''
    binary = os.path.join(stella_path(grid, precision, backend),
                          f'StandaloneStencils{suffix}')
    if not os.path.isfile(binary):
        raise RuntimeError(f'Could not find STELLA binary at {binary}')
    return binary


def stella_command(grid, precision, backend, stencil, size):
    binary = stella_binary(grid, precision, backend)
    ni, nj, nk = size
    filt = stencil.stella_filter
    return f'{binary} --ie {ni} --je {nj} --ke {nk} --gtest_filter={filt}'


def gridtools_path(grid, precision, backend):
    assert grid in {'strgrid', 'icgrid'}
    assert precision in {'float', 'double'}
    assert backend in {'host', 'cuda'}
    return system.gridtools_path(grid, precision, backend)


def gridtools_binary(grid, precision, backend, stencil):
    binary = getattr(stencil, 'gridtools_' + backend.lower())
    binary = os.path.join(gridtools_path(grid, precision, backend),
                          binary)
    if not os.path.isfile(binary):
        raise RuntimeError(f'Could not find GridTools binary at {binary}')
    return binary


def gridtools_command(grid, precision, backend, stencil, size):
    binary = gridtools_binary(grid, precision, backend, stencil)
    ni, nj, nk = size
    halo = stencil.halo
    ni, nj, nk = ni + 2 * halo, nj + 2 * halo, nk + 2 * halo
    return f'{binary} {ni} {nj} {nk} 10'


def command(grid, precision, runtime, backend, stencil, size):
    assert runtime in {'stella', 'gridtools'}
    com = gridtools_command if runtime == 'gridtools' else stella_command
    return com(grid, precision, backend, stencil, size)
