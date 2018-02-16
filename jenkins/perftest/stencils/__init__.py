# -*- coding: utf-8 -*-

import importlib
import re

from perftest import ConfigError, logger
from perftest.utils import Registry


class Stencil(metaclass=Registry):
    @property
    def name(self):
        clsname = self.__class__.__name__
        return re.sub(r'(.)([A-Z]+)', r'\1 \2', clsname).lower()


def stencils_module(grid):
    stencils = importlib.import_module('perftest.stencils.' + grid)
    logger.debug(f'Successfully imported stencils for grid "{grid}"')
    return stencils


def instantiate(grid):
    stencils = stencils_module(grid)
    return [cls() for cls in stencils.Stencil.registry]


def domains(grid):
    stencils = stencils_module(grid)
    return stencils.domains
