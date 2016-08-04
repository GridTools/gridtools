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

        # Create stencil
        self.dt_small = 10.0 / 3.0
        self.dlat = 0.02
        self.flat_limit = 11
        self.stencil = FastWavesUV (domain=self.domain,
                                    dt_small=self.dt_small,
                                    dlat=self.dlat,
                                    flat_limit=self.flat_limit)
        self.stencil.set_halo((3,3,3,3))
        self.stencil.set_k_direction('forward')
        self.stencil.set_backend ('python')

        # Stencil inputs
        self.u_pos = np.random.random(self.domain) / 1e2
        self.v_pos = np.random.random(self.domain) / 1e2
        self.utens_stage = np.random.random(self.domain) / 1e2
        self.vtens_stage = np.random.random(self.domain) / 1e2
        self.ppuv      = np.random.random(self.domain) / 1e2
        self.rho       = np.random.random(self.domain) / 1e2
        self.rho0      = np.random.random(self.domain) / 1e2
        self.p0        = np.random.random(self.domain) / 1e2
        self.hhl       = np.random.random(self.domain) / 1e2
        self.wgtfac    = np.random.random(self.domain) / 1e2
        self.fx        = np.random.random(self.domain) / 1e2
        self.cwp       = np.random.random(self.domain) / 1e2
        self.xdzdx     = np.random.random(self.domain) / 1e2
        self.xdzdy     = np.random.random(self.domain) / 1e2
        self.xlhsx     = np.random.random(self.domain) / 1e2
        self.xlhsy     = np.random.random(self.domain) / 1e2
        self.wbbctens_stage = np.random.random ((self.domain[0],self.domain[1],self.domain[2]+1)) / 1e2

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
        self.stencil.set_backend (backend)

        # Run stencil
        self._run()

        # Check results
        diff_detected = False
        for k in 'uv':
            diff  = np.isclose(eval('self.out_%s'%k)[3:-3,3:-3], eval('self.ref_%s'%k)[3:-3,3:-3], atol=1e-12)
            ndiff = np.count_nonzero (np.logical_not (diff))
            if ndiff:
                logging.debug("\tDifferences detected for parameter 'out_%s'" % k)
                reldiff = (np.abs(eval('self.%s'%k)[3:-3,3:-3] - eval('self.ref_%s'%k)[3:-3,3:-3])
                           / eval('self.ref_%s'%k)[3:-3,3:-3])
                logging.debug("\t\tNumber of differences: %d" % ndiff)
                logging.debug("\t\tMax relative difference: %.7g" % np.max(reldiff))
                logging.debug("\t\tMean relative difference: %.7g" % np.mean(reldiff))
                logging.debug("\t\tRelative difference stddev: %.7g" % np.std(reldiff))
                diff_detected = True

        self.assertFalse (diff_detected)


    @unittest.skip("To be validated")
    def test_stella_results_cpp (self):
        self.test_stella_results (backend='c++')


    @unittest.skip("To be validated")
    @attr(lang='cuda')
    def test_stella_results_cuda (self):
        self.test_stella_results (backend='cuda')


    def test_data_dependency_detection (self, deps=None, backend='c++'):
        expected_deps = [('out_u', 'u_pos'),
                         ('out_u', 'xlhsx'),
                         ('out_u', 'xdzdx'),
                         ('out_u', 'xdzdy'),
                         ('out_u', 'self.xrhsx'),
                         ('out_u', 'self.xrhsy'),
                         ('out_u', 'self.xrhsz'),
                         ('out_u', 'utens_stage'),
                         ('out_u', 'self.ppgradu'),
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
                         ('self.ppgradcor_init', 'wgtfac'),
                         ('self.ppgradcor_init', 'ppuv'),
                         ('self.ppgradcor', 'self.ppgradcor_init'),
                         ('self.ppgradcor', 'wgtfac'),
                         ('self.ppgradcor', 'ppuv')]

        super ( ).test_data_dependency_detection (deps=expected_deps,
                                                  backend=backend)

    @unittest.skip("To be implemented")
    @attr(lang='cuda')
    def test_data_dependency_detection_cuda (self, deps=None, backend='cuda'):
        pass


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS
        #
        # fields and their ranges
        #
        self.add_expected_offset ('wgtfac',  None)
        self.add_expected_offset ('wgtfac',  None)
        self.add_expected_offset ('ppuv',  [0,0,0,0])
        self.add_expected_offset ('ppuv',  [0,0,0,0])
        self.add_expected_offset ('ppuv',  None)
        self.add_expected_offset ('ppuv',  [0,1,0,0])
        self.add_expected_offset ('ppuv',  [0,1,0,1])
        self.add_expected_offset ('ppuv',  [0,0,0,1])
        self.add_expected_offset ('ppuv',  [0,1,0,1])
        self.add_expected_offset ('self.ppgradcor_init',  [0,0,0,0])
        self.add_expected_offset ('self.ppgradcor_init',  None)
        self.add_expected_offset ('self.ppgradcor',  None)
        self.add_expected_offset ('self.ppgradcor',  None)
        self.add_expected_offset ('self.ppgradcor',  [0,1,0,1])
        self.add_expected_offset ('self.ppgradu',  None)
        self.add_expected_offset ('self.ppgradu',  None)
        self.add_expected_offset ('self.ppgradu',  None)
        self.add_expected_offset ('self.ppgradv',  None)
        self.add_expected_offset ('self.ppgradv',  None)
        self.add_expected_offset ('self.ppgradv',  None)
        self.add_expected_offset ('self.xrhsx', None)
        self.add_expected_offset ('self.xrhsx', [-1,0,0,1])
        self.add_expected_offset ('self.xrhsy', None)
        self.add_expected_offset ('self.xrhsy', [0,1,-1,0])
        self.add_expected_offset ('self.xrhsz', None)
        self.add_expected_offset ('self.xrhsz', [0,1,0,1])
        self.add_expected_offset ('xlhsx', None)
        self.add_expected_offset ('xlhsy', None)
        self.add_expected_offset ('xdzdx', [-1,0,0,1])
        self.add_expected_offset ('xdzdy', [0,1,-1,0])
        self.add_expected_offset ('fx',   None)
        self.add_expected_offset ('fx',   None)
        self.add_expected_offset ('rho0', None)
        self.add_expected_offset ('rho', None)
        self.add_expected_offset ('rho', [0,1,0,0])
        self.add_expected_offset ('rho', [0,0,0,1])
        self.add_expected_offset ('rho', [0,1,0,1])
        self.add_expected_offset ('cwp',   None)
        self.add_expected_offset ('p0',   None)
        self.add_expected_offset ('hhl', [0,1,0,1])
        self.add_expected_offset ('utens_stage', None)
        self.add_expected_offset ('utens_stage', None)
        self.add_expected_offset ('vtens_stage', None)
        self.add_expected_offset ('vtens_stage', None)
        self.add_expected_offset ('wbbctens_stage', [0,0,0,0])
        self.add_expected_offset ('bott_u', None)
        self.add_expected_offset ('bott_v', None)
        self.add_expected_offset ('rhou', None)
        self.add_expected_offset ('rhov', None)
        self.add_expected_offset ('u_pos', None)
        self.add_expected_offset ('u_pos', None)
        self.add_expected_offset ('v_pos', None)
        self.add_expected_offset ('v_pos', None)
        self.add_expected_offset ('out_u', None)
        self.add_expected_offset ('out_u', None)
        self.add_expected_offset ('out_v', None)
        self.add_expected_offset ('out_v', None)
        self.add_expected_offset ('self.edadlat', None)
        self.add_expected_offset ('self.edadlat', None)
        self.add_expected_offset ('self.gravity', None)
        self.add_expected_offset ('self.dt_small', None)
        self.add_expected_offset ('self.dt_small', None)

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_compare_python_and_cpp_results (self, backend='c++'):
        params_shapes  = dict ( )
        for p in self.params:
            params_shapes[p]  = self.domain
        params_shapes['wbbctens_stage'] = (self.domain[0],self.domain[1],(self.domain[2]+1))
        super ( ).test_compare_python_and_cpp_results (params_shapes=params_shapes,
                                                       backend=backend)


    @attr(lang='cuda')
    def test_compare_python_and_cuda_results (self):
        self.test_compare_python_and_cpp_results (backend='cuda')


    def test_ghost_cell_pattern (self, expected_patterns=None, backend='c++'):
        if expected_patterns is None:
            expected_patterns = {'stage_ppgradcor_init':[0,1,0,1],
                                 'stage_ppgradcor_below_top':[0,1,0,1],
                                 'stage_ppgradcor_at_top':[0,1,0,1],
                                 'stage_xrhsx':[1,0,0,1],
                                 'stage_xrhsy':[0,1,1,0],
                                 'stage_xrhsz':[0,1,0,1],
                                 'stage_ppgrad_at_flat_limit':[0,0,0,0],
                                 'stage_ppgrad_over_flat_limit':[0,0,0,0],
                                 'stage_uv':[0,0,0,0],
                                 'stage_uv_boundary':[0,0,0,0]
                                }
        super ( ).test_ghost_cell_pattern (expected_patterns,
                                           backend=backend)


    def test_minimum_halo_detection (self, min_halo=None):
        if min_halo is None:
            min_halo = [1, 1, 1, 1]
        super ( ).test_minimum_halo_detection (min_halo)


    @unittest.skip("Superseded by tests on STELLA results")
    def test_python_results (self, out_param=None, result_file=None):
        pass


    @unittest.skip("To be verified")
    @attr(lang='cuda')
    def test_execution_performance_cuda (self):
        pass
