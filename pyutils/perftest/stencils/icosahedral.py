# -*- coding: utf-8 -*-

import os

from perftest.stencils import Stencil


def path(stencil):
    return os.path.join('regression', 'icosahedral', stencil)


class StencilOnEdgesMultifields(Stencil):
    gridtools_path = path('stencil_on_edges_multiplefields')
    halo = 1


class StencilOnCells(Stencil):
    gridtools_path = path('stencil_on_cells_path')
    halo = 1


class StencilOnNeighcellOfEdges(Stencil):
    gridtools_path = path('stencil_on_neighcell_of_edges')
    halo = 1


class StencilManualFold(Stencil):
    gridtools_path = path('stencil_manual_fold')
    halo = 1
