# -*- coding: utf-8 -*-

import importlib
import platform
import re


def name():
    hostname = platform.node()
    return re.sub(r'^([a-z]+)(ln-)?\d*$', '\g<1>', hostname)


system = importlib.import_module('perftest.machine.' + name())

StellaRuntime = system.StellaRuntime
GridtoolsRuntime = system.GridtoolsRuntime
sbatch = system.sbatch
