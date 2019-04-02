# -*- coding: utf-8 -*-

import os

from perftest.stencils import Stencil


def path(stencil):
    return os.path.join('regression', stencil)


class Copy(Stencil):
    stella_filter = 'BasicBenchmarks.copyStencilTest'
    gridtools_path = path('copy_stencil')
    halo = 0


class HorizontalDiffusion(Stencil):
    stella_filter = 'HoriDiffBenchmark.4Stages'
    gridtools_path = path('horizontal_diffusion')
    halo = 2


class HorizontalDiffusionFused(Stencil):
    stella_filter = 'HoriDiffBenchmark.4Stages'
    gridtools_path = path('horizontal_diffusion_fused')
    halo = 2


class SimpleHorizontalDiffusion(Stencil):
    stella_filter = 'HoriDiffBenchmark.SingleVar'
    gridtools_path = path('simple_hori_diff')
    halo = 2


class VerticalAdvection(Stencil):
    stella_filter = 'VerticalAdvectionBenchmark.U'
    gridtools_path = path('vertical_advection_dycore')
    halo = 3


class AdvectionPdBott(Stencil):
    stella_filter = 'AdvectionPDBottPrepareTracersBenchmark.MultipleTracers'
    gridtools_path = path('advection_pdbott_prepare_tracers')
    halo = 0
