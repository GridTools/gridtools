# -*- coding: utf-8 -*-

import importlib
import os
import platform
import re
import subprocess

from perftest import ConfigError, logger
from perftest.runtime import GridtoolsRuntime, Runtime


def hostname():
    """Host name of the current machine.

    Example:
        >>> hostname()
        'keschln-0002'
    """
    hostname = platform.node()
    logger.debug(f'Host name is {hostname}')
    return hostname


def clustername():
    """SLURM cluster name of the current machine.

    Examples:
        >>> clustername()
        'kesch'
    """
    output = subprocess.check_output(['scontrol', 'show', 'config']).decode()
    p = re.compile(r'.*ClusterName\s*=\s*(\S*).*', re.MULTILINE | re.DOTALL)
    m = p.match(output)
    if not m:
        raise ConfigError('Could not get SLURM cluster name')
    return m.group(1)


def get(config=None):
    """Loads a config with the given name or default system config.

    If the argument is an instance of the class `Config`, it is returned
    without modification.

    Args:
      config: The name of the config to load, a config object or `None`.

    Returns:
        A configuration represented by a `Config` object.
    """
    if isinstance(config, Config):
        return config

    if config is None:
        config = clustername()
    return Config(config)


class Config:
    """Main configuration class.

    Imports a config from a module and presents an interface to the config
    classes and functions.
    """
    def __init__(self, name):
        self.name = name
        self.hostname = hostname()
        self.clustername = clustername()

        logger.debug(f'Trying to load config "{self.name}"')
        self._config = importlib.import_module('perftest.config.' + self.name)
        logger.info(f'Successfully loaded config "{self.name}"')

        required_attrs = ['modules', 'env', 'cmake_command', 'make_command',
                          'sbatch']
        for attr in required_attrs:
            if not hasattr(self._config, attr):
                raise ConfigError(f'Incomplete config "{self.name}", missing '
                                  f'attribute "{attr}"')

    @property
    def env(self):
        if not hasattr(self, '_env'):
            self._env = os.environ.copy()
            if self._config.modules:
                from perftest import modules
                logger.debug(f'Trying to load modules for config '
                             f'"{self.name}"')
                for module in self._config.modules:
                    self._env = modules.load(self._env, module)
                logger.info(f'Successfully loaded modules for config '
                            f'"{self.name}"')
            self._env.update({str(k): str(v) for k, v in
                              self._config.env.items()})

            envstr = '\n'.join(f'{k}={v}' for k, v in self.env.items())
            logger.debug(f'Environment for config "{self.name}":', envstr)
        return self._env

    @property
    def make_command(self):
        return self._config.make_command

    @property
    def cmake_command(self):
        return self._config.cmake_command

    def runtime(self, runtime, *args, **kwargs):
        """Searches for and instantiates the given runtime with the arguments.

        Looks in the loaded config module for all classes derived from
        `perftest.runtime.Runtime` and checks if there is one with a name
        matching the given argument `runtime`. For comparison, the class name
        is converted to lowercase and the string "runtime" is removed. E.g.
        the argument "foo" will match a class FooRuntime.

        Args:
            runtime: Lower case name of the runtime to load.
            *args: Arguments passed on to the constructor of the runtime class.
            **kwargs: Keyword arguments, passed on to the runtime class.

        Returns:
            The instantiated runtime object.
        """
        logger.debug(f'Trying to get runtime "{runtime}"')
        if runtime == 'gridtools':
            return GridtoolsRuntime(self)

        for k, v in self._config.__dict__.items():
            if isinstance(v, type) and (issubclass(v, Runtime) and
                                        v is not Runtime):
                if v.__name__.lower().rstrip('runtime') == runtime:
                    return v(self, *args, **kwargs)
        raise ConfigError(f'Runtime "{runtime}" not available')

    def sbatch(self, command):
        """Generates a SLURM sbatch file to run the given `command`.

        Args:
            command: A command line command as a string.

        Returns:
            A string of the generated SLURM sbatch file.
        """
        return self._config.sbatch(command)
