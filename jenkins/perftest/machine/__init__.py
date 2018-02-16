# -*- coding: utf-8 -*-

import importlib
import platform
import re

from perftest import logger


def name():
    hostname = platform.node()
    logger.debug(f'Host name is {hostname}')
    machinename = re.sub(r'^([a-z]+)(ln-)?\d*$', '\g<1>', hostname)
    logger.debug(f'Machine name is {machinename}')
    return machinename


system = importlib.import_module('perftest.machine.' + name())

StellaRuntime = system.StellaRuntime
GridtoolsRuntime = system.GridtoolsRuntime
sbatch = system.sbatch

logger.debug(f'Successfully imported machine config')
