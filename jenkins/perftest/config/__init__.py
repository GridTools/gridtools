# -*- coding: utf-8 -*-

import importlib
import platform
import re

from perftest import ConfigError, logger


_config = None


def get_hostname():
    hostname = platform.node()
    logger.debug(f'Host name is {hostname}')
    return hostname


def get_systemname():
    systemname = re.sub(r'^([a-z]+)(ln-)?\d*$', '\g<1>', get_hostname())
    logger.debug(f'System name is {systemname}')
    return systemname


def load_config(configname=None):
    if configname is None:
        configname = get_systemname()

    from perftest.runtime import Runtime
    global _config

    logger.debug(f'Trying to load config "{configname}"')
    config_module = importlib.import_module('perftest.config.' + configname)
    config = dict()

    try:
        config['sbatch'] = config_module.sbatch
    except AttributeError:
        raise ConfigError(f'Loading config "{configname}" failed, '
                          'no sbatch function provided in config') from None

    config['runtimes'] = dict()
    for k, v in config_module.__dict__.items():
        if isinstance(v, type) and issubclass(v, Runtime):
            runtime_name = k.lower().rstrip('runtime')
            config['runtimes'][runtime_name] = v
            logger.debug(f'Found runtime "{runtime_name}" in config')

    config['name'] = configname

    _config = config
    logger.debug(f'Successfully loaded config "{configname}"')


def get_runtime(runtime):
    if not _config:
        load_config()
    try:
        return _config['runtimes'][runtime]
    except KeyError:
        configname = _config['name']
        clsname = runtime.title() + 'Runtime'
        raise ConfigError(f'Config "{configname}" does not provide '
                          f'a runtime "{runtime}" (class "{clsname}") in its '
                          'config file') from None


def get_configname():
    if not _config:
        load_config()
    return _config['name']


def get_sbatch(command):
    if not _config:
        load_config()
    return _config['sbatch'](command)
