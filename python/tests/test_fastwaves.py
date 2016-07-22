# -*- coding: utf-8 -*-
import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil  import Stencil, MultiStageStencil
from tests.test_stencils import CopyTest

from stella.fastwaves import FastWavesUV

#
# Prevent CopyTest test cases from running
#
CopyTest.__test__ = False



class FastWavesUVTest (CopyTest):
    """
    A test case for the STELLA FastWavesUV stencil defined in stella.stencils.
    """
    __test__ = True


    def setUp (self):
        super ( ).setUp( )
        self.domain = (32, 32, 80)
        self.params = ('u_pos',
                       'v_pos',
                       'out_u',
                       'out_v',
                       'utens_stage',
                       'vtens_stage',
                       'ppuv',
                       'rho',
                       'rho0',
                       'p0',
                       'hhl',
                       'wgtfac',
                       'fx',
                       'cwp',
                       'xdzdx',
                       'xdzdy',
                       'xlhsx',
                       'xlhsy',
                       'wbbctens_stage')
        self.temps  = ('self.xrhsx',
                       'self.xrhsy',
                       'self.xrhsz',
                       'self.ppgradcor_init',
                       'self.ppgradcor',
                       'self.ppgradu',
                       'self.ppgradv')

        self.stencil = FastWavesUV (self.domain)
        self.stencil.set_halo((3,3,3,3))
        self.stencil.set_k_direction('forward')

        # Stencil inputs
        self.u_pos = np.random.random(self.domain) / 1e-4
        self.v_pos = np.random.random(self.domain) / 1e-4
        self.utens_stage = np.random.random(self.domain) / 1e-4
        self.vtens_stage = np.random.random(self.domain) / 1e-4
        self.ppuv      = np.random.random(self.domain) / 1e-4
        self.rho       = np.random.random(self.domain) / 1e-4
        self.rho0      = np.random.random(self.domain) / 1e-4
        self.p0        = np.random.random(self.domain) / 1e-4
        self.hhl       = np.random.random(self.domain) / 1e-4
        self.wgtfac    = np.random.random(self.domain) / 1e-4
        self.fx        = np.random.random(self.domain) / 1e-4
        self.cwp       = np.random.random(self.domain) / 1e-4
        self.xdzdx     = np.random.random(self.domain) / 1e-4
        self.xdzdy     = np.random.random(self.domain) / 1e-4
        self.xlhsx     = np.random.random(self.domain) / 1e-4
        self.xlhsy     = np.random.random(self.domain) / 1e-4
        self.wbbctens_stage = np.random.random ((self.domain[0],self.domain[1],self.domain[2]+1)) / 1e-4

        # Stencil outputs
        self.out_u = np.zeros (self.domain, dtype=np.float64)
        self.out_v = np.zeros (self.domain, dtype=np.float64)

        dx, dy, dz = [ 1./i for i in self.domain ]
        for p in Stencil.get_interior_points(self.u_pos,
                                             ghost_cell=[0,0,0,0]):
            x = dx*p[0]
            y = dy*p[1]
            self.u_pos[p] = 0.01 + 0.19*(2.+np.cos(np.pi*(x+y)) + \
                                        np.sin(2*np.pi*(x+y)))/4.0
            self.v_pos[p] = 0.03 + 0.69*(2.+np.cos(np.pi*(x+y)) + \
                                        np.sin(2*np.pi*(x+y)))/4.0


    def test_stella_results (self, backend='python'):
        # The following test is run with input data originally taken from
        # /scratch/daint/jenkins/data/double/oldFW/
        # Results are compared with data coming from a standalone version of
        # the STELLA FastWavesUV benchmark stencil.
        # All input and output reference results are packaged in a compressed
        # NPZ archive.
        #
        # Reference output results were obtained with the following scalar
        # constants values:
        #
        self.dt_small = 10.0 / 3.0
        self.dlat = 0.02
        self.flat_limit = 11

        # Import data from NPZ compressed archive
        import os
        cur_dir = os.path.dirname (os.path.abspath (__file__))
        filename = "%s/FWdata_compressed.npz" % cur_dir
        with np.load(filename) as npz_data:
            # 3D stencil inputs
            self.u_pos       = npz_data['u_pos']
            self.v_pos       = npz_data['v_pos']
            self.utens_stage = npz_data['utens_stage']
            self.vtens_stage = npz_data['vtens_stage']
            self.ppuv        = npz_data['ppuv']
            self.rho         = npz_data['rho']
            self.wgtfac      = npz_data['wgtfac']

            # Single plane stencil inputs
            self.cwp            = npz_data['cwp']
            self.xdzdx          = npz_data['xdzdx']
            self.xdzdy          = npz_data['xdzdy']
            self.xlhsx          = npz_data['xlhsx']
            self.xlhsy          = npz_data['xlhsy']
            self.wbbctens_stage = npz_data['wbbctens_stage']

            # Constant field inputs
            self.rho0    = npz_data['rho0']
            self.p0      = npz_data['p0']
            self.hhl     = npz_data['hhl']
            self.acrlat0 = npz_data['acrlat0']

            # STELLA reference results
            self.ref_u = npz_data['ref_u']
            self.ref_v = npz_data['ref_v']

        # Reset domain
        self.domain = self.u_pos.shape

        # Initialize fx input field (not included in NPZ archive for clarity)
        self.eddlat = 180.0 / self.dlat / np.pi
        self.fx = self.eddlat * self.acrlat0
        self.fx = np.tile(self.fx, (self.domain[0], self.domain[2], 1)).swapaxes(1, 2)

        # Initialize stencil outputs
        self.out_u = np.zeros(self.domain, dtype=np.float64)
        self.out_v = np.zeros(self.domain, dtype=np.float64)

        # Reset stencil to use STELLA's data domain size
        self.stencil = FastWavesUV (domain=self.domain,
                                    dt_small=self.dt_small,
                                    dlat=self.dlat,
                                    flat_limit=self.flat_limit)
        self.stencil.set_halo((3,3,3,3))
        self.stencil.set_k_direction('forward')

        # Run stencil
        self._run()

        # Compare results
        udiff = np.isclose(self.out_u, self.ref_u, atol=1e-12)
        vdiff = np.isclose(self.out_v, self.ref_v, atol=1e-12)
        num_udiff = np.count_nonzero(np.logical_not (udiff))
        num_vdiff = np.count_nonzero(np.logical_not (vdiff))
        print ("Num udiff:", num_udiff)
        print ("Num vdiff:", num_vdiff)
        if num_udiff or num_vdiff:
            reldiff_u = (self.out_u - self.ref_u) / self.ref_u
            reldiff_v = (self.out_v - self.ref_v) / self.ref_v
            print ("mean udiff:", np.mean(reldiff_u))
            print ("mean vdiff:", np.mean(reldiff_v))
            print ("max udiff:", np.max(reldiff_u))
            print ("max vdiff:", np.max(reldiff_v))
            print ("stddev udiff:", np.std(reldiff_u))
            print ("stddev vdiff:", np.std(reldiff_v))
        self.assertEqual (num_udiff, 0)
        self.assertEqual (num_udiff, 0)


    def test_data_dependency_detection (self, deps=None, backend='python'):
        expected_deps = [('out_u', 'u_pos'),
                         ('out_u', 'xlhsx'),
                         ('out_u', 'xdzdx'),
                         ('out_u', 'xdzdy'),
                         ('out_u', 'self.xrhsx'),
                         ('out_u', 'self.xrhsy'),
                         ('out_u', 'self.xrhsz'),
                         ('out_u', 'utens_stage'),
                         ('out_u', 'ppgradu'),
                         ('out_u', 'fx'),
                         ('out_u', 'rho'),
                         ('out_v', 'v_pos'),
                         ('out_v', 'xlhsy'),
                         ('out_v', 'xdzdx'),
                         ('out_v', 'xdzdy'),
                         ('out_v', 'self.xrhsx'),
                         ('out_v', 'self.xrhsy'),
                         ('out_v', 'self.xrhsz'),
                         ('out_v', 'vtens_stage'),
                         ('out_v', 'self.ppgradv'),
                         ('out_v', 'rho'),
                         ('self.ppgradu', 'ppuv'),
                         ('self.ppgradu', 'self.ppgradcor'),
                         ('self.ppgradu', 'hhl'),
                         ('self.ppgradv', 'ppuv'),
                         ('self.ppgradv', 'self.ppgradcor'),
                         ('self.ppgradv', 'hhl'),
                         ('self.xrhsz', 'rho0'),
                         ('self.xrhsz', 'rho'),
                         ('self.xrhsz', 'cwp'),
                         ('self.xrhsz', 'p0'),
                         ('self.xrhsz', 'ppuv'),
                         ('self.xrhsz', 'wbbctens_stage'),
                         ('self.xrhsy', 'rho'),
                         ('self.xrhsy', 'ppuv'),
                         ('self.xrhsy', 'vtens_stage'),
                         ('self.xrhsx', 'fx'),
                         ('self.xrhsx', 'rho'),
                         ('self.xrhsx', 'ppuv'),
                         ('self.xrhsx', 'utens_stage'),
                         ('self.ppgradcor', 'wgtfac'),
                         ('self.ppgradcor', 'ppuv')]

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

if __name__ == '__main__':
    t = FastWavesUVTest()
    t.setUp()
    t.test_stella_results()
