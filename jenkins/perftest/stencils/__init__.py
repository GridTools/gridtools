# -*- coding: utf-8 -*-

import importlib
import re

from perftest import logger


class Stencil():
    @property
    def name(self):
        clsname = type(self).__name__
        return re.sub(r'(.)([A-Z]+)', r'\1 \2', clsname).lower()


def load(grid):
    logger.debug(f'Trying to import stencils for grid "{grid}"')
    mod = importlib.import_module('perftest.stencils.' + grid)

    stencils = []
    for k, v in mod.__dict__.items():
        if isinstance(v, type) and issubclass(v, Stencil) and v is not Stencil:
            stencils.append(v())

    sstr = ', '.join(f'"{s.name}"' for s in stencils)
    logger.info(f'Successfully imported stencils {sstr} for grid "{grid}"')
    return stencils
