# -*- coding: utf-8 -*-

import os

from pyutils import buildinfo, env, log, runtools


def _ctest(good_labels, bad_labels, verbose):
    command = ['ctest', '--output-on-failure']
    if good_labels:
        command += ['-L', good_labels]
    if bad_labels:
        command += ['-LE', bad_labels]
    if verbose:
        command.append('-VV')
    return command


def _run_nompi(label, verbose_ctest):
    log.info('Running non-MPI tests', label)
    output, = runtools.sbatch([_ctest(label, 'mpi', verbose_ctest)],
                              cwd=buildinfo.binary_dir)
    log.info('ctest unit test output', output)


def _run_mpi(verbose_ctest):
    log.info('Running MPI tests')
    output, = runtools.sbatch([_ctest('mpi', None, verbose_ctest)],
                              cwd=buildinfo.binary_dir,
                              use_srun=False,
                              use_mpi_config=True)
    log.info('ctest MPI test output', output)


def run(label, run_mpi_tests, verbose_ctest):
    _run_nompi(label, verbose_ctest)
    if run_mpi_tests:
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
