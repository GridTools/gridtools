# -*- coding: utf-8 -*-

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
