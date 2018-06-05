# -*- coding: utf-8 -*-

import importlib
import os
import re

from perftest import logger


class Stencil():
    """Base class for all stencils."""

    @property
    def name(self):
        """Lower case stencil name.

        Returns:
            The CamelCase class name transformed to a lower case, space
            separated string.
        """

        clsname = type(self).__name__
        return re.sub(r'(.)([A-Z]+)', r'\1 \2', clsname).lower()

    def gridtools_binary(self, backend):
        return getattr(self, 'gridtools_' + backend)

    def gridtools_target(self, backend):
        return os.path.basename(self.gridtools_binary(backend))


def load(grid):
    """Stencil loading functions.

    Loads all stencils for the given grid from the respective module.

    Args:
        grid: Name of the grid for which the stencils should be loaded.

    Returns:
        A list of all stencils provided for the given type.
    """

    logger.debug(f'Trying to import stencils for grid "{grid}"')
    mod = importlib.import_module('perftest.stencils.' + grid)

    stencils = []
    for k, v in mod.__dict__.items():
        if isinstance(v, type) and issubclass(v, Stencil) and v is not Stencil:
            stencils.append(v())

    sstr = ', '.join(f'"{s.name}"' for s in stencils)
    logger.info(f'Successfully imported stencils {sstr} for grid "{grid}"')
    return stencils
