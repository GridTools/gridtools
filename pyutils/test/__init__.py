# -*- coding: utf-8 -*-

import os

from pyutils import buildinfo, env, log, runtools


def _ctest(label, verbose):
    command = ['ctest', '--output-on-failure', '-L', label]
    if verbose:
        command.append('-VV')
    return command


def _run_nompi(verbose_ctest):
    log.info('Running non-MPI tests')
    outputs = runtools.sbatch([_ctest('unittest_*', verbose_ctest),
                               _ctest('regression_*', verbose_ctest)],
                              cwd=buildinfo.binary_dir)
    unit_exitcode, stdout, stderr = outputs[0]
    log.info('ctest unit test output', stdout)
    regression_exitcode, stdout, stderr = outputs[1]
    log.info('ctest regression test output', stdout)

    if unit_exitcode != 0 or regression_exitcode != 0:
        raise RuntimeError('ctest failed')


def _run_mpi(verbose_ctest):
    log.info('Running MPI tests')
    output, = runtools.sbatch([_ctest('mpitest_*', verbose_ctest)],
                              cwd=buildinfo.binary_dir, use_srun=False,
                              use_mpi_config=True)
    exitcode, stdout, stderr = output
    log.info('ctest MPI test output', stdout)

    if exitcode != 0:
        raise RuntimeError('ctest failed')


def run(mpi, verbose_ctest):
    _run_nompi(verbose_ctest)
    if mpi:
        _run_mpi(verbose_ctest)


def compile_examples(build_dir):
    import build
    from pyutils import buildinfo

    source_dir = os.path.join(buildinfo.install_dir, 'gridtools_examples')
    build_dir = os.path.abspath(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    env.set_cmake_arg('CMAKE_BUILD_TYPE', buildinfo.build_type.title())

    log.info('Configuring examples')
    build.cmake(source_dir, build_dir)
    log.info('Building examples')
    build.make(build_dir)
    log.info('Successfully built examples')
