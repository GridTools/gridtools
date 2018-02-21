# -*- coding: utf-8 -*-

import asyncio
import os
import re
import subprocess
import tempfile
import textwrap

from perftest import JobError, logger


def run(commands, sbatch_gen=None):
    if isinstance(commands, str):
        commands = [commands]
    futures = [asyncio.ensure_future(_run(c, sbatch_gen)) for c in commands]
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))

    return [future.result() for future in futures]


def _submit(command, sbatch_gen=None):
    if sbatch_gen is None:
        from perftest import config
        sbatch_gen = config.sbatch

    with tempfile.NamedTemporaryFile(suffix='.sh', mode='w') as sbatch:
        sbatchstr = sbatch_gen(command)
        logger.debug(f'Generated sbatch file:\n' +
                     textwrap.indent(sbatchstr, '    '))
        sbatch.write(sbatchstr)
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
        sacct_command = ['sacct', '--format=jobid,jobname,state,exitcode',
                         '--parsable2', '--noheader', '--jobs=' + str(task_id)]
        logger.debug('Running "{}"'.format(' '.join(sacct_command)))
        info = subprocess.check_output(sacct_command).decode().strip()
        if info:
            logger.debug(f'Sacct output while waiting for {task_id}:\n' +
                         textwrap.indent(info, '    '))

            infos = [line.split('|') for line in info.split('\n')]
            if not any(state in wait_states for _, _, state, _ in infos):
                break
        else:
            logger.debug(f'Sacct gave no output while waiting for {task_id}')

        await asyncio.sleep(1)

    logger.debug(f'Job {task_id} finished')

    failed = False
    for jobid, jobname, state, exitcode in infos:
        exitcode = int(exitcode.split(':')[0])
        if exitcode != 0:
            logger.error(f'Job {jobid} ({jobname}) failed with exitcode '
                         f'{exitcode} and state {state}')
            failed = True

    with open(outpath, 'r') as out:
        output = out.read()
    os.remove(outpath)

    if failed:
        raise JobError(f'Job {task_id} failed with output:\n' +
                       textwrap.indent(output, '    '))

    logger.debug(f'Job {task_id} generated output:\n' +
                 textwrap.indent(output, '    '))

    return output


async def _run(command, sbatch_template=None):
    task_id, outpath = _submit(command, sbatch_template)
    return await _wait(task_id, outpath)
