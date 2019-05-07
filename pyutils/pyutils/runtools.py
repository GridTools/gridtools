# -*- coding: utf-8 -*-

import os
import re
import statistics
import subprocess
import tempfile
import time

from pyutils import env, log


def run(command, **kwargs):
    if not command:
        raise ValueError('No command provided')

    log.info('Invoking', ' '.join(command))
    start = time.time()
    try:
        output = subprocess.check_output(command,
                                         env=env.env,
                                         stderr=subprocess.STDOUT,
                                         **kwargs)
    except subprocess.CalledProcessError as e:
        log.error(f'{command[0]} failed with output', e.output.decode())
        raise e
    end = time.time()
    log.info(f'{command[0]} finished in {end - start:.2f}s')
    output = output.decode().strip()
    log.debug(f'{command[0]} output', output)
    return output


def _sbatch_file(rundir):
    return os.path.join(rundir, 'run.sh')


def _stdout_file(rundir, command_id):
    return os.path.join(rundir, f'stdout_{command_id}.out')


def _stderr_file(rundir, command_id):
    return os.path.join(rundir, f'stderr_{command_id}.out')


def _generate_sbatch(commands, cwd, use_srun, use_mpi_config):
    code = f'#!/bin/bash -l\n#SBATCH --array=0-{len(commands) - 1}\n'

    for option in env.sbatch_options(use_mpi_config):
        code += f'#SBATCH {option}\n'

    if cwd is None:
        cwd = os.path.abspath(os.getcwd())
    srun = env.srun_command() if use_srun else ''

    code += f'cd {cwd}\n'
    code += 'case $SLURM_ARRAY_TASK_ID in\n'
    for i, command in enumerate(commands):
        commandstr = ' '.join(command)
        code += f'    {i})\n        {srun} {commandstr}\n        ;;\n'
    code += '    *)\nesac'
    return code


def _run_sbatch(rundir, commands, cwd, use_srun, use_mpi_config):
    sbatchstr = _generate_sbatch(commands, cwd, use_srun, use_mpi_config)
    log.debug('Generated sbatch file', sbatchstr)
    with open(_sbatch_file(rundir), 'w') as sbatch:
        sbatch.write(sbatchstr)

    command = ['sbatch',
               '--output', _stdout_file(rundir, '%a'),
               '--error', _stderr_file(rundir, '%a'),
               '--wait',
               _sbatch_file(rundir)]
    log.info('Invoking sbatch', ' '.join(command))
    start = time.time()
    result = subprocess.run(command,
                            env=env.env,
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    end = time.time()
    log.info(f'sbatch finished in {end - start:.2f}s')
    if result.returncode != 0:
        log.warning(f'sbatch finished with exit code '
                    f'{result.returncode} and message',
                    result.stderr.decode())
    return int(re.match(r'Submitted batch job (\d+)',
                        result.stdout.decode()).group(1))


def _retreive_outputs(rundir, commands, task_id):
    command = ['sacct',
               '--jobs', f'{task_id}',
               '--format', 'jobid,exitcode',
               '--parsable2',
               '--noheader']
    for i in range(1, 7):
        try:
            output = run(command)
        except subprocess.CalledProcessError:
            time.sleep(1)
            continue
        infos = [o.split('|') for o in output.splitlines() if '.batch' in o]
        exitcodes = [int(code.split(':')[0]) for _, code in sorted(infos)]
        if len(exitcodes) == len(commands):
            break
        time.sleep(i**2)
    else:
        raise RuntimeError('Could not get exit codes of jobs')

    time.sleep(5)

    outputs = []
    for i, (command, exitcode) in enumerate(zip(commands, exitcodes)):
        if exitcode != 0:
            log.debug(f'Exit code of command "{command}"', exitcode)
        with open(_stdout_file(rundir, i), 'r') as outfile:
            stdout = outfile.read()
            if stdout.strip():
                log.debug(f'Stdout of command "{command}"', stdout)
        with open(_stderr_file(rundir, i), 'r') as outfile:
            stderr = outfile.read()
            if stderr.strip():
                log.debug(f'Stderr of command "{command}"', stderr)
        outputs.append((exitcode, stdout, stderr))
    return outputs


def sbatch(commands, cwd=None, use_srun=True, use_mpi_config=False):
    with tempfile.TemporaryDirectory(dir='.') as rundir:
        task = _run_sbatch(rundir, commands, cwd, use_srun, use_mpi_config)
        return _retreive_outputs(rundir, commands, task)


def sbatch_retry(commands, retries, *args, **kwargs):
    outputs = sbatch(commands, *args, **kwargs)
    for retry in range(retries):
        exitcodes = [exitcode for exitcode, *_ in outputs]
        if all(exitcode == 0 for exitcode in exitcodes):
            break
        if statistics.mode(exitcodes) != 0:
            raise RuntimeError('Majority of jobs has failed')

        failed_commands = []
        failed_indices = []
        for i, (command, output) in enumerate(zip(commands, outputs)):
            exitcode, *_ = output
            if exitcode != 0:
                failed_commands.append(command)
                failed_indices.append(i)

        failed_outputs = sbatch(failed_commands, *args, **kwargs)

        for i, o in zip(failed_indices, failed_outputs):
            outputs[i] = o
    for command, (exitcode, stdout, stderr) in zip(commands, outputs):
        if exitcode != 0:
            raise RuntimeError(f'Command "{command}" still failed after '
                               f'{retries} retries with output: {stderr}')
    return outputs
