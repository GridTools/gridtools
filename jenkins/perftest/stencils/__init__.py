# -*- coding: utf-8 -*-

import importlib
import re

from perftest import ConfigError
from perftest.helpers import Registry


class Stencil(metaclass=Registry):
    def __init__(self):
        required_attrs = ['stella_filter', 'gridtools_cuda',
                          'gridtools_host', 'halo']
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(self, attr):
                missing_attrs.append(attr)
        if missing_attrs:
            missing = ' '.join(missing_attrs)
            raise ConfigError(f'Stencil {self.name} has missing attributes: {missing}')

    @property
    def name(self):
        return re.sub(r'(.)([A-Z]+)', r'\1 \2', self.__class__.__name__).lower()


def instantiate(grid):
    stencils = importlib.import_module('perftest.stencils.' + grid)
    return [cls() for cls in stencils.Stencil.registry]


def sizes(grid):
    stencils = importlib.import_module('perftest.stencils.' + grid)
    return stencils.sizes
