# -*- coding: utf-8 -*-

from pyutils import log, runtools


def _ctest(label):
    return f'ctest --output_on_failure . -L "{label}"'


def _run_nompi(env):
    sbatch_options = env.sbatch_options(mpi=False)
    srun = env.srun_command()
    outputs = runtools.run(env, [_ctest('unittest_*'), _ctest('regression_*')],
                           sbatch_options, srun)
    unit_exitcode, stdout, stderr = outputs[0]
    log.info('ctest unit test output', stdout)
    regression_exitcode, stdout, stderr = outputs[1]
    log.info('ctest regression test output', stdout)

    if unit_exitcode != 0 or regression_exitcode != 0:
        raise RuntimeError('ctest failed')


def _run_mpi(env):
    sbatch_options = env.sbatch_options(mpi=True)
    srun = ''
    output, = runtools.run(env, [_ctest('mpitest_*')], sbatch_options, srun)
    exitcode, stdout, stderr = output
    log.info('ctest MPI test output', stdout)

    if exitcode != 0:
        raise RuntimeError('ctest failed')


def run(env, mpi):
    _run_nompi(env)
    if mpi:
        _run_mpi(env)
