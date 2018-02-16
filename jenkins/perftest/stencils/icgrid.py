# -*- coding: utf-8 -*-

from perftest.stencils import Stencil


domains = [(16, 16, 8), (32, 32, 8)]


def path(stencil):
    return os.path.join('examples', 'icosahedral', stencil)


class StencilOnEdgesMultifields(Stencil):
    gridtools_gpu = path('stencil_on_edges_multiplefields_cuda')
    gridtools_host = path('stencil_on_edges_multiplefields_host_block')
    halo = 1


class StencilOnCells(Stencil):
    gridtools_gpu = path('stencil_on_cells_cuda')
    gridtools_host = path('stencil_on_cells_host_block')
    halo = 1


class StencilOnNeighcellOfEdges(Stencil):
    gridtools_gpu = path('sencil_on_neighcell_of_edges_cuda')
    gridtools_host = path('sencil_on_neighcell_of_edges_host_block')
    halo = 1

class StencilManualFold(Stencils):
    gridtools_gpu = path('stencil_manual_fold_cuda')
    gridtools_host = path('stencil_manual_fold_host_block')
    halo = 1
