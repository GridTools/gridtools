# -*- coding: utf-8 -*-

import importlib
import platform
import re

from perftest import ConfigError, logger


imported_configname = None
imported_runtimes = None
imported_sbatch = None


def system_name():
    hostname = platform.node()
    logger.debug(f'Host name is {hostname}')
    machinename = re.sub(r'^([a-z]+)(ln-)?\d*$', '\g<1>', hostname)
    logger.debug(f'Machine name is {machinename}')
    return machinename


def load_config(configname):
    from perftest.runtime import Runtime
    global imported_configname, imported_runtimes, imported_sbatch

    logger.debug(f'Trying to load config "{configname}"')

    config_module = importlib.import_module('perftest.config.' + configname)

    try:
        imported_sbatch = config_module.sbatch
    except AttributeError:
        raise ConfigError(f'Loading config "{configname}" failed, '
                          'no sbatch function provided in config') from None

    imported_runtimes = dict()
    for k, v in config_module.__dict__.items():
        if isinstance(v, type) and issubclass(v, Runtime):
            runtime_name = k.lower().rstrip('runtime')
            imported_runtimes[runtime_name] = v
            logger.debug(f'Found runtime "{runtime_name}" in config '
                         '"{configname}"')

    imported_configname = configname

    logger.debug(f'Successfully imported config "{configname}"')


def get_runtime(runtime):
    if not imported_runtimes:
        raise ConfigError('No config was loaded')
    try:
        return imported_runtimes[runtime]
    except KeyError:
        clsname = runtime.title() + 'Runtime'
        raise ConfigError(f'Config "{imported_configname}" does not provide '
                          f'a runtime "{runtime}" (class "{clsname}") in its '
                          'config file') from None


def sbatch(command):
    if not imported_sbatch:
        raise ConfigError('No config was loaded')
    return imported_sbatch(command)
