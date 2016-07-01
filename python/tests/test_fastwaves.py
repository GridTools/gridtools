# -*- coding: utf-8 -*-
import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil  import Stencil, MultiStageStencil
from tests.test_stencils import CopyTest

from stella.stencil import FastWavesUV



class FastWavesUVTest (CopyTest):
    """
    A test case for the STELLA FastWavesUV stencil defined in stella.stencils.
    """
    def setUp (self):
        super ( ).setUp( )
        # self.domain = (128, 128, 64)
        self.domain = (16, 16, 8)
        self.params = ('in_u',
                       'in_v',
                       'out_u',
                       'out_v')
        self.temps  = ('self.utens_stage',
                       'self.vtens_stage',
                       'self.u_pos',
                       'self.v_pos',
                       'self.ppuv',
                       'self.rho',
                       'self.rho0',
                       'self.p0',
                       'self.hhl',
                       'self.wgtfac',
                       'self.fx',
                       'self.cwp',
                       'self.xdzdx',
                       'self.xdzdy',
                       'self.xlhsx',
                       'self.xlhsy',
                       'self.wbbctens_stage',
                       'self.xrhsx',
                       'self.xrhsy',
                       'self.xrhsz',
                       'self.ppgradcor',
                       'self.ppgradu',
                       'self.ppgradv')

        self.stencil = FastWavesUV (self.domain, halo=(3,3,3,3))

        self.in_u = np.zeros (self.domain, dtype=np.float64)
        self.in_v = np.zeros (self.domain, dtype=np.float64)
        self.out_u = np.zeros (self.domain, dtype=np.float64)
        self.out_v = np.zeros (self.domain, dtype=np.float64)

        dx, dy, dz = [ 1./i for i in self.domain ]
        for p in self.stencil.get_interior_points(self.in_u,
                                                  ghost_cell=[-3,3,-3,3]):
            x = dx*p[0]
            y = dy*p[1]
            self.in_u[p] = 0.01 + 0.19*(2.+np.cos(np.pi*(x+y)) + \
                                        np.sin(2*np.pi*(x+y)))/4.0
            self.in_v[p] = 0.03 + 0.69*(2.+np.cos(np.pi*(x+y)) + \
                                        np.sin(2*np.pi*(x+y)))/4.0

        # self.stencil.plot_data_dependency(graph=None,
        #                                   outfile='fastwaves_dep.pdf')


    def test_data_dependency_detection (self, deps=None, backend='python'):
        expected_deps = [('out_u', 'self.u_pos'),
                         ('out_u', 'self.xlhsx'),
                         ('out_u', 'self.xdzdx'),
                         ('out_u', 'self.xdzdy'),
                         ('out_u', 'self.xrhsx'),
                         ('out_u', 'self.xrhsy'),
                         ('out_u', 'self.xrhsz'),
                         ('out_u', 'self.utens_stage'),
                         ('out_u', 'self.ppgradu'),
                         ('out_u', 'self.fx'),
                         ('out_u', 'self.rho'),
                         ('out_v', 'self.v_pos'),
                         ('out_v', 'self.xlhsy'),
                         ('out_v', 'self.xdzdx'),
                         ('out_v', 'self.xdzdy'),
                         ('out_v', 'self.xrhsx'),
                         ('out_v', 'self.xrhsy'),
                         ('out_v', 'self.xrhsz'),
                         ('out_v', 'self.vtens_stage'),
                         ('out_v', 'self.ppgradv'),
                         ('out_v', 'self.rho'),
                         ('self.ppgradu', 'self.ppuv'),
                         ('self.ppgradu', 'self.ppgradcor'),
                         ('self.ppgradu', 'self.hhl'),
                         ('self.ppgradv', 'self.ppuv'),
                         ('self.ppgradv', 'self.ppgradcor'),
                         ('self.ppgradv', 'self.hhl'),
                         ('self.xrhsz', 'self.rho0'),
                         ('self.xrhsz', 'self.rho'),
                         ('self.xrhsz', 'self.cwp'),
                         ('self.xrhsz', 'self.p0'),
                         ('self.xrhsz', 'self.ppuv'),
                         ('self.xrhsz', 'self.wbbctens_stage'),
                         ('self.xrhsy', 'self.rho'),
                         ('self.xrhsy', 'self.ppuv'),
                         ('self.xrhsy', 'self.vtens_stage'),
                         ('self.xrhsx', 'self.fx'),
                         ('self.xrhsx', 'self.rho'),
                         ('self.xrhsx', 'self.ppuv'),
                         ('self.xrhsx', 'self.utens_stage'),
                         ('self.ppgradcor', 'self.wgtfac'),
                         ('self.ppgradcor', 'self.ppuv'),
                         ('self.u_pos', 'in_u'),
                         ('self.v_pos', 'in_v')]

        super ( ).test_data_dependency_detection (deps=expected_deps,
                                                  backend=backend)


    def test_data_dependency_detection_cuda (self, deps=None, backend='cuda'):
        pass


    def test_automatic_access_pattern_detection (self):
        pass


    def test_compare_python_cpp_and_cuda_results (self):
        pass


    def test_ghost_cell_pattern (self, expected_patterns=None, backend='c++'):
        pass


    def test_minimum_halo_detection (self, min_halo=[0,0,0,0]):
        pass


    def test_symbol_discovery (self, backend='c++'):
        pass


    def test_symbol_discovery_cuda (self):
        pass


    def test_user_stencil_extends_multistagestencil (self):
        pass


    def test_kernel_function (self):
        pass


    def test_run_stencil_only_accepts_keyword_arguments (self):
        pass


    def test_python_results (self, out_param=None, result_file=None):
        pass


    def test_execution_performance_cpp (self, backend='c++'):
        pass


    def test_execution_performance_cuda (self):
        pass


    def test_k_directions (self, backend='c++'):
        pass


    def test_k_directions_cuda (self):
        pass
