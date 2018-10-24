# -*- coding: utf-8 -*-

import os

from perftest.stencils import Stencil


def path(stencil):
    return os.path.join('regression', 'icosahedral', stencil)


class StencilOnEdgesMultifields(Stencil):
    gridtools_cuda = path('stencil_on_edges_multiplefields_cuda')
    gridtools_x86= path('stencil_on_edges_multiplefields_x86_block')
    gridtools_mc = path('stencil_on_edges_multiplefields_mc')
    halo = 1


class StencilOnCells(Stencil):
    gridtools_cuda = path('stencil_on_cells_cuda')
    gridtools_x86= path('stencil_on_cells_x86_block')
    gridtools_mc = path('stencil_on_cells_mc')
    halo = 1


class StencilOnNeighcellOfEdges(Stencil):
    gridtools_cuda = path('stencil_on_neighcell_of_edges_cuda')
    gridtools_x86= path('stencil_on_neighcell_of_edges_x86_block')
    gridtools_mc = path('stencil_on_neighcell_of_edges_mc')
    halo = 1


class StencilManualFold(Stencil):
    gridtools_cuda = path('stencil_manual_fold_cuda')
    gridtools_x86= path('stencil_manual_fold_x86_block')
    gridtools_mc = path('stencil_manual_fold_mc')
    halo = 1
