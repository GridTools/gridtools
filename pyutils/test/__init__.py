# -*- coding: utf-8 -*-

import os
import subprocess
import time

from pyutils import buildinfo, log, runtools


def _ctest(label, verbose):
    command = f'ctest --output-on-failure -L "{label}"'
    if verbose:
        command += ' -VV'
    return command


def _run_nompi(env, verbose_ctest):
    sbatch_options = env.sbatch_options(mpi=False)
    srun = env.srun_command()
    outputs = runtools.run(env,
                           [_ctest('unittest_*', verbose_ctest),
                            _ctest('regression_*', verbose_ctest)],
                           sbatch_options,
                           srun,
                           cwd=buildinfo.binary_dir)
    unit_exitcode, stdout, stderr = outputs[0]
    log.info('ctest unit test output', stdout)
    regression_exitcode, stdout, stderr = outputs[1]
    log.info('ctest regression test output', stdout)

    if unit_exitcode != 0 or regression_exitcode != 0:
        raise RuntimeError('ctest failed')


def _run_mpi(env, verbose_ctest):
    sbatch_options = env.sbatch_options(mpi=True)
    srun = ''
    output, = runtools.run(env, [_ctest('mpitest_*', verbose_ctest)],
                           sbatch_options, srun, cwd=buildinfo.binary_dir)
    exitcode, stdout, stderr = output
    log.info('ctest MPI test output', stdout)

    if exitcode != 0:
        raise RuntimeError('ctest failed')


def run(env, mpi, verbose_ctest):
    _run_nompi(env, verbose_ctest)
    if mpi:
        _run_mpi(env, verbose_ctest)


def compile_examples(env, build_dir):
    import build
    from pyutils import buildinfo

    source_dir = os.path.join(buildinfo.install_dir, 'gridtools_examples')
    build_dir = os.path.abspath(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    env.set_cmake_arg('CMAKE_BUILD_TYPE', buildinfo.build_type.title())
    env.set_cmake_arg('GT_EXAMPLES_FORCE_CUDA', buildinfo.target == 'gpu')

    build.cmake(env, source_dir, build_dir)
    build.make(env, build_dir, build_command=env.build_command())
