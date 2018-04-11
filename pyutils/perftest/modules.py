# -*- coding: utf-8 -*-

import re
import subprocess

from perftest import ConfigError, logger


def _apply_modulecmd_output(env, output):
    p_assign = re.compile("os.environ\['(\w*)'\] = '(.*)'")
    p_delete = re.compile("del os.environ\['(\w*)'\]")

    env = dict(env)
    for line in output.splitlines():
        m_assign = p_assign.match(line)
        m_delete = p_delete.match(line)
        if m_assign:
            env[m_assign.group(1)] = m_assign.group(2)
        elif m_delete:
            del env[m_delete.group(1)]
        else:
            raise ConfigError(f'Could not parse modulecmd output: "{line}"')
    return env


def _call_modulecmd(env, *modulecmd_args):
    output = subprocess.check_output(('modulecmd', 'python') + modulecmd_args,
                                     stderr=subprocess.STDOUT, env=env)
    return output.decode().strip()


def unload(env, module):
    logger.debug(f'Unloading module "{module}"')
    output = _call_modulecmd(env, 'unload', module)
    return _apply_modulecmd_output(env, output)


def swap(env, swapout_module, swapin_module):
    logger.debug(f'Swapping out module "{swapout_module}" in favor of '
                 f'module "{swapin_module}"')
    output = _call_modulecmd(env, 'swap', swapout_module, swapin_module)
    return _apply_modulecmd_output(env, output)


def load(env, module, resolve_conflicts=True):
    output = _call_modulecmd(env, 'load', module)
    if 'conflict' in output:
        conflicting = []
        p = re.compile(r'.*Tcl command execution failed: conflict (.*)')
        for line in output.splitlines():
            m = p.match(line)
            if m:
                conflicting.append(m.group(1))
        confstr = ', '.join(f'"{c}"' for c in conflicting)
        logger.debug(f'Module "{module}" conflicts with {confstr}')
        if not resolve_conflicts:
            raise ConfigError(f'Unresolved conflicts between module "{module}"'
                              f' and modules {confstr}')
        else:
            logger.debug('Resolving module conflicts…')
            if len(conflicting) == 1:
                return swap(env, conflicting[0], module)
            else:
                for c in conflicting:
                    env = unload(env, c)
                return load(env, module, resolve_conflicts=False)
    else:
        return _apply_modulecmd_output(env, output)
