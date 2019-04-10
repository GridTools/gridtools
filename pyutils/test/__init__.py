# -*- coding: utf-8 -*-

from pyutils import Error, log, runtools


class TestFailure(Error):
    pass


def run(env):
    test_env = env.copy()
    test_env['GTRUN_SRUN_COMMAND'] = ''

    output, = runtools.run(test_env,
                           [f'ctest --output-on-failure . -L "unittest_*"'])
    exitcode, stdout, stderr = output

    log.info('ctest output', stdout)

    if exitcode != 0:
        raise TestFailure('ctest failed')
