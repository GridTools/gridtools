# -*- coding: utf-8 -*-

import os

from perftest.stencils import Stencil


def path(stencil):
    return os.path.join('regression', stencil)


class Copy(Stencil):
    stella_filter = 'BasicBenchmarks.copyStencilTest'
    gridtools_cuda = path('copy_stencil_cuda')
    gridtools_x86= path('copy_stencil_x86_block')
    gridtools_mc = path('copy_stencil_mc')
    halo = 0


class HorizontalDiffusion(Stencil):
    stella_filter = 'HoriDiffBenchmark.4Stages'
    gridtools_cuda = path('interface1_cuda')
    gridtools_x86= path('interface1_x86_block')
    gridtools_mc = path('interface1_mc')
    halo = 2


class HorizontalDiffusionFused(Stencil):
    stella_filter = 'HoriDiffBenchmark.4Stages'
    gridtools_cuda = path('interface1_fused_cuda')
    gridtools_x86= path('interface1_fused_x86_block')
    gridtools_mc = path('interface1_fused_mc')
    halo = 2


class SimpleHorizontalDiffusion(Stencil):
    stella_filter = 'HoriDiffBenchmark.SingleVar'
    gridtools_cuda = path('simple_hori_diff_cuda')
    gridtools_x86= path('simple_hori_diff_x86_block')
    gridtools_mc = path('simple_hori_diff_mc')
    halo = 2


class VerticalAdvection(Stencil):
    stella_filter = 'VerticalAdvectionBenchmark.U'
    gridtools_cuda = path('vertical_advection_dycore_cuda')
    gridtools_x86= path('vertical_advection_dycore_x86_block')
    gridtools_mc = path('vertical_advection_dycore_mc')
    halo = 3


class AdvectionPdBott(Stencil):
    stella_filter = 'AdvectionPDBottPrepareTracersBenchmark.MultipleTracers'
    gridtools_cuda = path('advection_pdbott_prepare_tracers_cuda')
    gridtools_x86= path('advection_pdbott_prepare_tracers_x86_block')
    gridtools_mc = path('advection_pdbott_prepare_tracers_mc')
    halo = 0
