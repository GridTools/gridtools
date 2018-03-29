# -*- coding: utf-8 -*-

import asyncio
import os
import random
import re
import subprocess
import tempfile
import time

from perftest import config, JobError, logger


def run(commands, conf=None, job_limit=None):
    """Runs the given command(s) using SLURM and the given configuration.

    `conf` must be a valid argument for `perftest.config.get`.

    Args:
        commands: A string or a list of strings, console command(s) to run.
        conf:  (Default value = None) The config to use or None for default.
        job_limit: An integer defining the max number of jobs submitted to
                   SLURM in parallel.

    Returns:
        A list of collected console outputs of all commands.
    """
    if isinstance(commands, str):
        commands = [commands]

    conf = config.get(conf)

    if job_limit is None:
        job_limit = len(commands)

    # Initialize list of outputs
    outputs = [None] * len(commands)

    # Put all commands into an asyncio queue
    queue = asyncio.Queue()
    for i, command in enumerate(commands):
        queue.put_nowait((i, command))

    # executor consumer coroutine, gets a command from the queue and runs it
    async def executor():
        while True:
            i, command = await queue.get()
            outputs[i] = await _run(command, conf)
            queue.task_done()

    # main execution coroutine, runs `job_limit` number of executors and waits
    # until the queue is empty
    async def execute():
        executors = [asyncio.ensure_future(executor()) for _
                     in range(job_limit)]
        await queue.join()
        for e in executors:
            e.cancel()

    # start execution
    asyncio.get_event_loop().run_until_complete(execute())
    return outputs


def _submit(command, conf):
    """Submits a command to SLURM using sbatch."""

    with tempfile.NamedTemporaryFile(suffix='.sh', mode='w') as sbatch:
        # Generate SLURM sbatch file contents to submit job
        sbatchstr = conf.sbatch(command)
        logger.debug(f'Generated sbatch file:', sbatchstr)
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
        sbatch_out = subprocess.check_output(sbatch_command, env=conf.env)

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
        # Allow other tasks to run
        await asyncio.sleep(0)

        # Wait for job to finish (we use time.sleep instead asyncio.sleep
        # because we do not want to poll SLURM too often asynchronously)
        time.sleep(1)

        # Run sacct to get job status
        sacct_command = ['sacct', '--format=jobid,jobname,state,exitcode',
                         '--parsable2', '--noheader', '--jobs=' + str(task_id)]
        logger.debug('Running "{}"'.format(' '.join(sacct_command)))
        info = subprocess.check_output(sacct_command).decode().strip()
        if info:
            # Parse sacct output
            logger.debug(f'Sacct output while waiting for {task_id}:', info)
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
        raise JobError(f'Job {task_id} failed with output:', output)

    logger.debug(f'Job {task_id} generated output:', output)

    return output


async def _run(command, conf):
    """Asynchronous run command."""
    task_id, outpath = _submit(command, conf)
    return await _wait(task_id, outpath)
