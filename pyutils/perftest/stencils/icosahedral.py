# -*- coding: utf-8 -*-

import os

from perftest.stencils import Stencil


def path(stencil):
    return os.path.join('regression', 'icosahedral', stencil)


class StencilOnEdgesMultifields(Stencil):
    gridtools_cuda = path('stencil_on_edges_multiplefields_cuda')
    gridtools_host = path('stencil_on_edges_multiplefields_host_block')
    gridtools_mc = path('stencil_on_edges_multiplefields_mc')
    halo = 1


class StencilOnCells(Stencil):
    gridtools_cuda = path('stencil_on_cells_cuda')
    gridtools_host = path('stencil_on_cells_host_block')
    gridtools_mc = path('stencil_on_cells_mc')
    halo = 1


class StencilOnNeighcellOfEdges(Stencil):
    gridtools_cuda = path('stencil_on_neighcell_of_edges_cuda')
    gridtools_host = path('stencil_on_neighcell_of_edges_host_block')
    gridtools_mc = path('stencil_on_neighcell_of_edges_mc')
    halo = 1


class StencilManualFold(Stencil):
    gridtools_cuda = path('stencil_manual_fold_cuda')
    gridtools_host = path('stencil_manual_fold_host_block')
    gridtools_mc = path('stencil_manual_fold_mc')
    halo = 1
