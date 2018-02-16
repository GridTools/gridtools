# -*- coding: utf-8 -*-

import asyncio
import os
import re
import subprocess
import tempfile
import textwrap

from perftest import logger


def run(commands, sbatch_gen=None):
    if isinstance(commands, str):
        commands = [commands]
    futures = [asyncio.ensure_future(_run(c, sbatch_gen)) for c in commands]
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))

    return [future.result() for future in futures]


def _submit(command, sbatch_gen=None):
    if sbatch_gen is None:
        import perftest.machine as machine
        sbatch_gen = machine.sbatch

    with tempfile.NamedTemporaryFile(suffix='.sh', mode='w') as sbatch:
        sbatch.write(sbatch_gen(command))
        sbatch.flush()

        out = tempfile.NamedTemporaryFile(suffix='.out', dir='.', delete=False)
        out.close()

        sbatch_command = ['sbatch', '-o', out.name, sbatch.name]
        sbatch_out = subprocess.check_output(sbatch_command)

        task_id = re.match(r'Submitted batch job (\d+)',
                           sbatch_out.decode()).group(1)

    logger.debug(f'Submitted job {task_id}: "{command}"')

    return task_id, out.name


async def _wait(task_id, outpath):
    wait_states = {'PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING'}
    while True:
        sacct_command = ['sacct', '--format=state,exitcode', '--parsable2',
                         '--noheader', '--jobs=' + str(task_id)]
        logger.debug('Running "{}"'.format(' '.join(sacct_command)))
        info = subprocess.check_output(sacct_command).decode().strip()
        if info:
            state, exitcode = info.split('\n')[0].split('|')
            logger.debug(f'Sacct output while waiting for {task_id}:\n' +
                         textwrap.indent(info, '    '))
            if state not in wait_states:
                break
        else:
            logger.debug(f'Sacct gave no output while waiting for {task_id}')

        await asyncio.sleep(1)
    exitcode = int(exitcode.split(':')[0])
    logger.debug(f'Job {task_id} finished with exitcode {exitcode}')

    with open(outpath, 'r') as out:
        output = out.read()
    os.remove(outpath)

    logger.debug(f'Job {task_id} generated output:\n' +
                textwrap.indent(output, '    '))

    return output, exitcode


async def _run(command, sbatch_template=None):
    task_id, outpath = _submit(command, sbatch_template)
    output, exitcode = await _wait(task_id, outpath)
    if exitcode != 0:
        raise RuntimeError(f'Running command "{command}" failed with output:\n'
                           + output)
    return output
