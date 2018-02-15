# -*- coding: utf-8 -*-

from perftest.stencils import Stencil


sizes = [(16, 16, 8)]


class Copy(Stencil):
    stella_filter = 'BasicBenchmarks.copyStencilTest'
    gridtools_cuda = 'examples/copy_stencil_cuda'
    gridtools_host = 'examples/copy_stencil_host_block'
    halo = 0


class HorizontalDiffusion(Stencil):
    stella_filter = 'HoriDiffBenchmark.4Stages'
    gridtools_cuda = 'examples/interface1_cuda'
    gridtools_host = 'examples/interface1_host_block'
    halo = 2


class SimpleHorizontalDiffusion(Stencil):
    stella_filter = 'HoriDiffBenchmark.SingleVar'
    gridtools_cuda = 'examples/simple_hori_diff_cuda'
    gridtools_host = 'examples/simple_hori_diff_host_block'
    halo = 2


class VerticalAdvection(Stencil):
    stella_filter = 'VerticalAdvectionBenchmark.U'
    gridtools_cuda = 'examples/vertical_advection_dycore_cuda'
    gridtools_host = 'examples/vertical_advection_dycore_host_block'
    halo = 3


class AdvectionPdBott(Stencil):
    stella_filter = 'AdvectionPDBottPrepareTracersBenchmark.MultipleTracers'
    gridtools_cuda = 'examples/advection_pdbott_prepare_tracers_cuda'
    gridtools_host = 'examples/advection_pdbott_prepare_tracers_host_block'
    halo = 0
