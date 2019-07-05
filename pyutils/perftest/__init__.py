# -*- coding: utf-8 -*-

import os
import re

from pyutils import env, log, runtools
from perftest import stencils as stencil_loader
from perftest import result, time


def _stencil_binary(backend, stencil):
    from pyutils import buildinfo
    binary = os.path.join(buildinfo.binary_dir,
                          stencil.gridtools_binary(backend))
    if not os.path.isfile(binary):
        raise FileNotFoundError(f'Could not find GridTools binary "{binary}"')
    return binary


def _stencil_command(backend, stencil, domain):
    binary = _stencil_binary(backend, stencil)
    ni, nj, nk = domain
    halo = stencil.halo
    ni, nj = ni + 2 * halo, nj + 2 * halo
    return [binary, str(ni), str(nj), str(nk), '10']


def _git_commit():
    from pyutils import buildinfo
    return runtools.run(['git', 'rev-parse', 'HEAD'], cwd=buildinfo.source_dir)


def _git_datetime():
    from pyutils import buildinfo
    posixtime = runtools.run(
            ['git', 'show', '-s', '--format=%ct', _git_commit()],
            cwd=buildinfo.source_dir)
    return time.from_posix(posixtime)


def _parse_time(output):
    p = re.compile(r'.*\[s\]\s*([0-9.]+).*', re.MULTILINE | re.DOTALL)
    m = p.match(output)
    if not m:
        raise RuntimeError(f'Could not parse time in output:\n{output}')
    return float(m.group(1))


def run(domain, runs):
    from pyutils import buildinfo
    stencils = stencil_loader.load(buildinfo.grid)

    results = dict()
    for backend in buildinfo.backends:
        if backend == 'naive':
            continue

        commands = [_stencil_command(backend, s, domain) for s in stencils]
        allcommands = [c for c in commands for _ in range(runs)]
        log.info('Running stencils')
        alloutputs = runtools.sbatch_retry(allcommands, 5)
        log.info('Running stencils finished')
        alltimes = [_parse_time(o) for o in alloutputs]
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
