# -*- coding: utf-8 -*-

import os

from perftest.stencils import Stencil


def path(stencil):
    return os.path.join('examples', stencil)


class Copy(Stencil):
    stella_filter = 'BasicBenchmarks.copyStencilTest'
    gridtools_cuda = path('copy_stencil_cuda')
    gridtools_host = path('copy_stencil_host_block')
    gridtools_mic = path('copy_stencil_mic')
    halo = 0


class HorizontalDiffusion(Stencil):
    stella_filter = 'HoriDiffBenchmark.4Stages'
    gridtools_cuda = path('interface1_cuda')
    gridtools_host = path('interface1_host_block')
    gridtools_mic = path('interface1_mic')
    halo = 2


class HorizontalDiffusionFused(Stencil):
    stella_filter = 'HoriDiffBenchmark.4Stages'
    gridtools_cuda = path('interface1_fused_cuda')
    gridtools_host = path('interface1_fused_host_block')
    gridtools_mic = path('interface1_fused_mic')
    halo = 2


class SimpleHorizontalDiffusion(Stencil):
    stella_filter = 'HoriDiffBenchmark.SingleVar'
    gridtools_cuda = path('simple_hori_diff_cuda')
    gridtools_host = path('simple_hori_diff_host_block')
    gridtools_mic = path('simple_hori_diff_mic')
    halo = 2


class VerticalAdvection(Stencil):
    stella_filter = 'VerticalAdvectionBenchmark.U'
    gridtools_cuda = path('vertical_advection_dycore_cuda')
    gridtools_host = path('vertical_advection_dycore_host_block')
    gridtools_mic = path('vertical_advection_dycore_mic')
    halo = 3


class AdvectionPdBott(Stencil):
    stella_filter = 'AdvectionPDBottPrepareTracersBenchmark.MultipleTracers'
    gridtools_cuda = path('advection_pdbott_prepare_tracers_cuda')
    gridtools_host = path('advection_pdbott_prepare_tracers_host_block')
    gridtools_mic = path('advection_pdbott_prepare_tracers_mic')
    halo = 0
