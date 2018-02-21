# -*- coding: utf-8 -*-

import importlib
import platform
import re

from perftest import ConfigError, logger


def system_name():
    hostname = platform.node()
    logger.debug(f'Host name is {hostname}')
    machinename = re.sub(r'^([a-z]+)(ln-)?\d*$', '\g<1>', hostname)
    logger.debug(f'Machine name is {machinename}')
    return machinename


def load_config(name):
    global StellaRuntime, GridtoolsRuntime, sbatch

    logger.debug(f'Trying to load config "{name}"')

    system = importlib.import_module('perftest.config.' + name)
    StellaRuntime = system.StellaRuntime
    GridtoolsRuntime= system.GridtoolsRuntime
    sbatch = system.sbatch

    logger.debug(f'Successfully imported config "{name}"')


try:
    load_config(system_name())
except ModuleNotFoundError:
    logger.warn(f'Could not find default config for host "{system_name()}"')

    class StellaRuntime:
        def __init__(self, *args, **kwargs):
            raise ConfigError('No config was loaded')

    class GridtoolsRuntime:
        def __init__(self, *args, **kwargs):
            raise ConfigError('No config was loaded')

    def sbatch(*args, **kwargs):
        raise ConfigError('No config was loaded')
