# -*- coding: utf-8 -*-

import importlib
import platform
import re

from perftest import ConfigError, logger
from perftest.runtime import Runtime


def hostname():
    hostname = platform.node()
    logger.debug(f'Host name is {hostname}')
    return hostname


def systemname():
    systemname = re.sub(r'^([a-z]+)(ln-)?\d*$', '\g<1>', hostname())
    logger.debug(f'System name is {systemname}')
    return systemname


class Config:
    def __init__(self, name):
        self.name = name
        self.hostname = hostname()
        self.systemname = systemname()

        if self.name is None:
            self.name = self.systemname

        logger.debug(f'Trying to load config "{self.name}"')
        self._config = importlib.import_module('perftest.config.' + self.name)
        logger.info(f'Successfully loaded config "{self.name}"')

    def runtime(self, runtime, *args, **kwargs):
        logger.debug(f'Trying to get runtime "{runtime}"')
        for k, v in self._config.__dict__.items():
            if isinstance(v, type) and (issubclass(v, Runtime) and
                                        v is not Runtime):
                if v.__name__ == runtime.title() + 'Runtime':
                    return v(self, *args, **kwargs)
        raise ConfigError(f'Runtime "{runtime}" not available')

    def sbatch(self, command):
        return self._config.sbatch(command)


def load(config):
    return Config(config)
