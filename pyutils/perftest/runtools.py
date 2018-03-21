# -*- coding: utf-8 -*-

import asyncio
import os
import random
import re
import subprocess
import tempfile
import textwrap
import time

from perftest import JobError, logger


def run(commands, config=None):
    """Runs the given command(s) using SLURM.

    `config` must be a `perftest.config.Config` object or None, in which case
    the default configuration for the current system is loaded.

    Args:
        commands: A string or a list of strings, console command(s) to run.
        config:  (Default value = None) The config to use or None for default.

    Returns:
        A list of collected console outputs of all commands.
    """
    if isinstance(commands, str):
        commands = [commands]

    if config is None:
        import perftest.config
        config = perftest.config.load(config)

    futures = [asyncio.ensure_future(_run(c, config)) for c in commands]
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))

    return [future.result() for future in futures]


def _submit(command, config):
    """Submits a command to SLURM using sbatch."""

    with tempfile.NamedTemporaryFile(suffix='.sh', mode='w') as sbatch:
        # Generate SLURM sbatch file contents to submit job
        sbatchstr = config.sbatch(command)
        logger.debug(f'Generated sbatch file:\n' +
                     textwrap.indent(sbatchstr, '    '))
        # Write sbatch to file
        sbatch.write(sbatchstr)
        sbatch.flush()

        # Wait a bit to make sure that we don’t overload SLURM
        time.sleep(0.1)

        # Create a file to store the job output
        # It is created in the working dir (normal /tmp does not seem to work)
        out = tempfile.NamedTemporaryFile(suffix='.out', dir='.', delete=False)
        out.close()

        # Run sbatch to start the job and specify job output file
        sbatch_command = ['sbatch', '-o', out.name, sbatch.name]
        sbatch_out = subprocess.check_output(sbatch_command, env=config.env)

        # Parse the task ID from the sbatch stdout
        task_id = re.match(r'Submitted batch job (\d+)',
                           sbatch_out.decode()).group(1)

    logger.debug(f'Submitted job {task_id}: "{command}"')

    return task_id, out.name


async def _wait(task_id, outpath):
    """Asynchronously waits for a running SLURM job by polling."""

    # SLURM job states for unfinished jobs
    wait_states = {'PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING'}
    while True:
        # Wait for job to finish, randomized sleep times are used to minimize
        # the risk of polling SLURM too often (for different jobs, as for now
        # polling is done per job)
        await asyncio.sleep(random.uniform(5, 15))

        # Run sacct to get job status
        sacct_command = ['sacct', '--format=jobid,jobname,state,exitcode',
                         '--parsable2', '--noheader', '--jobs=' + str(task_id)]
        logger.debug('Running "{}"'.format(' '.join(sacct_command)))
        info = subprocess.check_output(sacct_command).decode().strip()
        if info:
            # Parse sacct output
            logger.debug(f'Sacct output while waiting for {task_id}:\n' +
                         textwrap.indent(info, '    '))
            infos = [line.split('|') for line in info.split('\n')]
            # Break out of loop if all jobs have finished
            if not any(state in wait_states for _, _, state, _ in infos):
                break
        else:
            # We do nothing (apart from logging) if sacct gives no output
            # Happens normally a very short time after job start
            logger.debug(f'Sacct gave no output while waiting for {task_id}')

    logger.debug(f'Job {task_id} finished')

    # Parse the final states and job return codes
    failed = False
    for jobid, jobname, state, exitcode in infos:
        exitcode = int(exitcode.split(':')[0])
        # Set `failed` flag if one subtask has non-zero exit code
        if exitcode != 0:
            logger.error(f'Job {jobid} ({jobname}) failed with exitcode '
                         f'{exitcode} and state {state}')
            failed = True

    # Try to read job output (independent of the job status)
    with open(outpath, 'r') as out:
        output = out.read()
    os.remove(outpath)

    # Raise error if job has failed, include the job output in the message
    if failed:
        raise JobError(f'Job {task_id} failed with output:\n' +
                       textwrap.indent(output, '    '))

    logger.debug(f'Job {task_id} generated output:\n' +
                 textwrap.indent(output, '    '))

    return output


async def _run(command, sbatch_template=None):
    """Asynchronous run command."""
    task_id, outpath = _submit(command, sbatch_template)
    return await _wait(task_id, outpath)
