# -*- coding: utf-8 -*-

import os
import re
import subprocess
import tempfile
import time

from perftest import config, JobError, JobSchedulingError, logger


def run(commands, conf=None, job_limit=None, retry=5):
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

    # Put all commands into a list with indices
    commands = list(enumerate(commands))

    # Set of running jobs
    running = set()

    while any(output is None for output in outputs):

        # Submit jobs if less than `job_limit` are running
        while len(running) < job_limit and commands:
            index, command = commands.pop()
            # Try to submit job `retry` times
            for _ in range(retry):
                try:
                    task_id, outfile = _submit(command, conf)
                    break
                except JobSchedulingError:
                    time.sleep(1)
            else:
                # Raise error if all attempts failed
                raise JobSchedulingError(f'Failed to run command "{command}", '
                                         f'all {retry} attempts failed')
            running.add((index, task_id, outfile))
            # Wait a bit to avoid overloading SLURM
            time.sleep(0.1)

        # Poll SLURM to get finished jobs
        try:
            finished_ids = _poll([task_id for _, task_id, _ in running])
        except JobError as e:
            # Cancel all (possibly still running) jobs
            for _, task_id, _ in running:
                subprocess.call(['scancel', '--full', str(task_id)],
                                env=conf.env)
            # Wait a few more seconds for buffered output
            time.sleep(10)
            # Print all current job outputs on job failure
            for _, task_id, outfile in running:
                with open(outfile, 'r') as f:
                    output = f.read()
                logger.debug(f'Current output of job {task_id}:', output)
                os.remove(outfile)
            # Re-raise job error
            raise e

        logger.debug('Finished jobs: ' + ', '.join(finished_ids))
        time.sleep(10)

        # Get output of finished jobs, build set of still running jobs
        still_running = set()
        for index, task_id, outfile in running:
            if task_id in finished_ids:
                logger.debug(f'Reading output of job {task_id}')
                with open(outfile, 'r') as f:
                    output = f.read()
                logger.debug(f'Job {task_id} generated output:', output)
                os.remove(outfile)
                outputs[index] = output
            else:
                still_running.add((index, task_id, outfile))
        running = still_running
        logger.debug('Running jobs: ' + ', '.join(task_id for _, task_id, _
                                                  in running))

    return outputs


def _submit(command, conf):
    """Submits a command to SLURM using sbatch.

    Args:
        command: A string, the command that should be submitted to SLURM.
        conf: The config to use for the submission.

    Returns:
        A tuple of the SLURM job ID and a temporary file name to which the
        job output will be written.
    """

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
        logger.debug(f'Created temporary output file {out.name} for command '
                     f'"{command}"')
        out.close()

        # Run sbatch to start the job and specify job output file
        sbatch_command = ['sbatch', '-o', out.name, '--requeue', sbatch.name]
        try:
            sbatch_out = subprocess.check_output(sbatch_command, env=conf.env)
        except subprocess.CalledProcessError as e:
            raise JobSchedulingError(f'Submitting job "{command}" failed '
                                     f'with output: {e.output}')

        # Parse the task ID from the sbatch stdout
        task_id = re.match(r'Submitted batch job (\d+)',
                           sbatch_out.decode()).group(1)

    logger.debug(f'Submitted job {task_id}: "{command}"')

    return task_id, out.name


def _poll(task_ids):
    """Polls SLURM using sacct.

    Args:
        task_ids: A list or set of SLURM job IDs.

    Returns:
        A subset of `task_ids` with all job IDs of jobs that have finished
        running.
    """

    # Early exit in case of empty job list
    if not task_ids:
        return set()

    # Run sacct to get job status
    jobstr = ','.join(task_ids)
    sacct_command = ['sacct', '--format=jobid,jobname,state,exitcode',
                     '--parsable2', '--noheader', '--jobs=' + jobstr]
    logger.debug('Running "{}"'.format(' '.join(sacct_command)))
    info = subprocess.check_output(sacct_command).decode().strip()

    finished = set()

    if not info:
        # We do nothing (apart from logging) if sacct gives no output
        # Happens normally a very short time after job start
        logger.debug(f'Sacct gave no output while waiting')
        return finished

    logger.debug('Sacct output while waiting:', info)

    # SLURM job states for unfinished jobs
    wait_states = {'PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING'}

    # Parse sacct output
    for line in info.splitlines():
        jobid, jobname, state, exitcode = line.split('|')
        if state not in wait_states:
            if state == 'CANCELLED':
                raise JobError(f'Job {jobid} ({jobname}) was cancelled')
            elif state == 'COMPLETED':
                # There might be additional internal subtasks in the output,
                # that we do not want to add to the set of finished jobs so we
                # have to check the ID here
                if jobid in task_ids:
                    finished.add(jobid)
            elif state == 'FAILED':
                exitcode = int(exitcode.split(':')[0])
                raise JobError(f'Job {jobid} ({jobname}) failed with exitcode '
                               f'{exitcode}')
            elif state == 'NODE_FAIL':
                # Ignore node failures, job should be automatically rescheduled
                # by SLURM here as we use the --requeue flag on submission
                pass
            elif state == 'TIMEOUT':
                raise JobError(f'Job {jobid} ({jobname}) timed out, consider '
                               f'increasing the time limit')
            else:
                raise JobError(f'Job {jobid} ({jobname}) failed with state '
                               f'{state}')

    return finished
