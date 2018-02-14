# -*- coding: utf-8 -*-

from perftest.config.stencils import Stencil


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
    halo = 0
