# -*- coding: utf-8 -*-

from pyutils import log, runtools


def run(env):
    test_env = env.copy()
    test_env['GTRUN_SRUN_COMMAND'] = ''

    output, = runtools.run(test_env, [f'ctest --output-on-failure .'])
    exitcode, stdout, stderr = output

    log.info('ctest output', stdout)

    if exitcode != 0:
        raise RuntimeError('ctest failed')
