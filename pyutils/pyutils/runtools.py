# -*- coding: utf-8 -*-

import os
import re
import statistics
import subprocess
import tempfile
import time

from pyutils import log


def _sbatch_file(rundir):
    return os.path.join(rundir, 'run.sh')


def _stdout_file(rundir, command_id):
    return os.path.join(rundir, f'stdout_{command_id}.out')


def _stderr_file(rundir, command_id):
    return os.path.join(rundir, f'stderr_{command_id}.out')


def _generate_sbatch(env, commands):
    code = f'#!/bin/bash -l\n#SBATCH --array=0-{len(commands) - 1}\n'

    settings = env.run_settings()

    for k, v in settings.items():
        if k.startswith('SBATCH_'):
            arg = '--' + k[7:].lower().replace('_', '-') + '=' + v
            code += f'#SBATCH {arg}\n'

    srun = settings.get('SRUN_COMMAND', 'srun')
    code += 'case $SLURM_ARRAY_TASK_ID in\n'
    for i, command in enumerate(commands):
        code += f'    {i})\n        {srun} {command}\n        ;;\n'
    code += '    *)\nesac'
    return code


def _run_sbatch(env, rundir, commands):
    sbatchstr = _generate_sbatch(env, commands)
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
                            env=env,
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


def _retreive_outputs(env, rundir, commands, task_id):
    command = ['sacct',
               '--jobs', f'{task_id}',
               '--format', 'jobid,exitcode',
               '--parsable2',
               '--noheader']
    for _ in range(10):
        log.info('Invoking sacct', ' '.join(command))
        output = subprocess.check_output(command, env=env).decode().strip()
        log.debug('sacct output', output)
        infos = [o.split('|') for o in output.splitlines() if '.batch' in o]
        exitcodes = [int(code.split(':')[0]) for _, code in sorted(infos)]
        if len(exitcodes) == len(commands):
            break
        time.sleep(1)

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


def run(env, commands):
    with tempfile.TemporaryDirectory(dir='.') as rundir:
        task = _run_sbatch(env, rundir, commands)
        return _retreive_outputs(env, rundir, commands, task)

def run_retry(env, commands, retries):
    outputs = run(env, commands)
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

        failed_outputs = run(env, failed_commands)

        for i, o in zip(failed_indices, failed_outputs):
            outputs[i] = o
    for command, (exitcode, stdout, stderr) in zip(commands, outputs):
        if exitcode != 0:
            raise RuntimeError(f'Command "{command}" still failed after '
                               f'{retries} retries with output: {stderr}')
    return outputs
