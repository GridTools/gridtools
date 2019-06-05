# -*- coding: utf-8 -*-

import os

from perftest.stencils import Stencil


def path(stencil):
    return os.path.join('regression', stencil)


class Copy(Stencil):
    gridtools_path = path('copy_stencil')
    halo = 0


class HorizontalDiffusion(Stencil):
    gridtools_path = path('horizontal_diffusion')
    halo = 2


class HorizontalDiffusionFused(Stencil):
    gridtools_path = path('horizontal_diffusion_fused')
    halo = 2


class SimpleHorizontalDiffusion(Stencil):
    gridtools_path = path('simple_hori_diff')
    halo = 2


class VerticalAdvection(Stencil):
    gridtools_path = path('vertical_advection_dycore')
    halo = 3


class AdvectionPdBott(Stencil):
    gridtools_path = path('advection_pdbott_prepare_tracers')
    halo = 0


class LayoutTransformation(Stencil):
    gridtools_path = path('layout_transformation')
    halo = 2

class BoundaryConditions(Stencil):
    gridtools_path = path('boundary_conditions')
    halo = 3
