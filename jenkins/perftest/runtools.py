#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import os
import re
import subprocess
import tempfile


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

    print(f'Submitted job {task_id}: "{command}"')

    return task_id, out.name


async def _wait(task_id, outpath):
    wait_states = {'PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING'}
    while True:
        info = subprocess.check_output(['sacct', '--format=state,exitcode',
                                        '--parsable2', '--noheader',
                                        '--jobs=' + str(task_id)]).decode()
        if info:
            state, exitcode = info.split('\n')[0].split('|')
            if state not in wait_states:
                break
        await asyncio.sleep(1)
    exitcode = int(exitcode.split(':')[0])

    with open(outpath, 'r') as out:
        output = out.read()
    os.remove(outpath)

    return output, exitcode


async def _run(command, sbatch_template=None):
    task_id, outpath = _submit(command, sbatch_template)
    output, exitcode = await _wait(task_id, outpath)
    if exitcode != 0:
        raise RuntimeError(f'Running command "{command}" failed with output:\n{output}')
    return output
