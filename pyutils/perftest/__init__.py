# -*- coding: utf-8 -*-

import collections
import os
import re
import subprocess

from pyutils import logger, runtools, ParseError, NotFoundError
from perftest import stencils as stencil_loader
from perftest import buildinfo, result, time


def _stencil_binary(backend, stencil):
    binary = os.path.join(buildinfo.binary_dir,
                          stencil.gridtools_binary(backend))
    if not os.path.isfile(binary):
        raise NotFoundError(f'Could not find GridTools binary at "{binary}"')
    return binary


def _stencil_command(backend, stencil, domain):
    binary = _stencil_binary(backend, stencil)
    ni, nj, nk = domain
    halo = stencil.halo
    ni, nj = ni + 2 * halo, nj + 2 * halo
    return f'{binary} {ni} {nj} {nk} 10'


def _git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=buildinfo.source_dir).decode().strip()


def _git_datetime():
    posixtime = subprocess.check_output(
            ['git', 'show', '-s', '--format=%ct', _git_commit()],
            cwd=buildinfo.source_dir).decode().strip()
    return time.from_posix(posixtime)


def _parse_time(output):
    p = re.compile(r'.*\[s\]\s*([0-9.]+).*', re.MULTILINE | re.DOTALL)
    m = p.match(output)
    if not m:
        raise ParseError(f'Could not parse time in output:\n{output}')
    return float(m.group(1))


def run(env, domain, runs):
    stencils = stencil_loader.load(buildinfo.grid)

    results = dict()
    for backend in buildinfo.backends:
        commands = [_stencil_command(backend, s, domain) for s in stencils]
        allcommands = [c for c in commands for _ in range(runs)]
        logger.info('Running stencils')
        alloutputs = runtools.run_retry(env, allcommands, 5)
        logger.info('Running stencils finished')
        alltimes = [_parse_time(o) for _, o, _ in alloutputs]
        times = [alltimes[i:i + runs] for i in range(0, len(alltimes), runs)]

        info = result.RunInfo(name='gridtools',
                              version=_git_commit(),
                              datetime=_git_datetime(),
                              precision=buildinfo.precision,
                              backend=backend,
                              grid=buildinfo.grid,
                              compiler=buildinfo.compiler,
                              hostname=env.hostname(),
                              clustername=env.clustername())

        results[backend] = result.from_data(info, domain, stencils, times)
    return results
