# -*- coding: utf-8 -*-

import importlib

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
        return self.__class__.__name__


def instantiate(grid_type):
    stencils = importlib.import_module('perftest.config.stencils.' + grid_type)
    return [cls() for cls in stencils.Stencil.registry]


def sizes(grid_type):
    stencils = importlib.import_module('perftest.config.stencils.' + grid_type)
    return stencils.sizes
