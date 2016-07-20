# -*- coding: utf-8 -*-
import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

import gridtools
from gridtools.stencil import Stencil, MultiStageStencil
from tests.test_stencils import CopyTest

#
# Prevent CopyTest test cases from running
#
CopyTest.__test__ = False


class GridToolsDataDeps (MultiStageStencil):
    """
    Replicates the first example of the Data Dependency Analysis page from the
    GridTools wiki:
    https://github.com/eth-cscs/gridtools/wiki/Data-Dependencies-Analysis-in-GridTools
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.a = np.zeros (domain)
        self.d = np.zeros (domain)


    def stage_f0 (self, a, b, c):
        for p in self.get_interior_points (a):
            a[p] = b[p + (-1,0,0)] + b[p + (1,0,0)] + c[p + (1,0,0)]


    def stage_f1 (self, b, c, d):
        for p in self.get_interior_points (d):
            d[p] = b[p + (-2,0,0)] + c[p + (-1,0,0)] + c[p + (2,0,0)]


    def stage_f2 (self, a, c, d, e):
        for p in self.get_interior_points (a):
            e[p] = (a[p + (-1,0,0)] + a[p + (2,0,0)]
                    + d[p + (-2,0,0)] + d[p + (2,0,0)]
                    + c[p + (-1,0,0)] + c[p + (1,0,0)])


    @Stencil.kernel
    def kernel (self, in_b, in_c, out_e):
        self.stage_f0 (a=self.a,
                       b=in_b,
                       c=in_c)

        self.stage_f1 (b=in_b,
                       c=in_c,
                       d=self.d)

        self.stage_f2 (a=self.a,
                       c=in_c,
                       d=self.d,
                       e=out_e)



class GridToolsDataDepsTest (CopyTest):
    """
    A test fixture for the GridToolsDataDeps stencil defined above.

    Contains a specific test case to validate individual access extents of
    stencil's input data fields
    """
    __test__ = True


    def setUp (self):
        super ( ).setUp ( )
        logging.basicConfig (level=logging.INFO)
        self.domain = (16, 16, 1)
        self.params = ('in_b', 'in_c', 'out_e')
        self.temps  = ('self.a', 'self.d')

        self.in_b  = np.ones (self.domain)
        self.in_c  = np.ones (self.domain)
        self.a = np.zeros (self.domain)
        self.d = np.zeros (self.domain)
        self.out_e = np.zeros (self.domain)

        # print (self.in_b, self.in_c)
        self.stencil = GridToolsDataDeps (self.domain)
        self.stencil.set_halo ( (4,4,0,0) )
        self.stencil.set_k_direction ('forward')


    def test_data_dependency_detection (self, expected_deps=None, backend='c++'):
        if expected_deps is None:
            expected_deps = [('out_e', 'in_c'),
                             ('out_e', 'self.d'),
                             ('out_e', 'self.a'),
                             ('self.d', 'in_b'),
                             ('self.d', 'in_c'),
                             ('self.a', 'in_b'),
                             ('self.a', 'in_c')]
        super ( ).test_data_dependency_detection (deps=expected_deps,
                                                  backend=backend)


    @attr(lang='cuda')
    def test_data_dependency_detection_cuda (self):
        self.test_data_dependency_detection (backend='cuda')


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS
        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_b', [-1,1,0,0])
        self.add_expected_offset ('in_b', [-2,0,0,0])
        self.add_expected_offset ('in_c', [0,1,0,0])
        self.add_expected_offset ('in_c', [-1,2,0,0])
        self.add_expected_offset ('in_c', [-1,1,0,0])
        self.add_expected_offset ('out_e', None)
        self.add_expected_offset ('self.a', None)
        self.add_expected_offset ('self.a', [-1,2,0,0])
        self.add_expected_offset ('self.d', None)
        self.add_expected_offset ('self.d', [-2,2,0,0])

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_ghost_cell_pattern (self, expected_patterns=None, backend='c++'):
        if expected_patterns is None:
            expected_patterns = {'stage_f0':[1,2,0,0],
                                 'stage_f1':[2,2,0,0],
                                 'stage_f2':[0,0,0,0] }
        super ( ).test_ghost_cell_pattern (expected_patterns,
                                           backend=backend)


    @attr(lang='cuda')
    def test_ghost_cell_pattern_cuda (self):
        self.test_ghost_cell_pattern (backend='cuda')


    def test_minimum_halo_detection (self, min_halo=None):
        if min_halo is None:
            min_halo = [4, 4, 0, 0]
        super ( ).test_minimum_halo_detection (min_halo)


    @unittest.skip("Not yet implemented")
    @attr(lang='python')
    def test_python_results (self):
        pass


    def test_parameter_access_extents (self):
        self._run ( )

        in_b = self.stencil.scope.symbol_table['in_b']
        in_c = self.stencil.scope.symbol_table['in_c']

        self.assertEqual (in_b.access_extent, [-4,3,0,0])
        self.assertEqual (in_c.access_extent, [-3,4,0,0])



class GridToolsDataDeps2 (MultiStageStencil):
    """
    Replicates the second example of the Data Dependency Analysis page from the
    GridTools wiki:
    https://github.com/eth-cscs/gridtools/wiki/Data-Dependencies-Analysis-in-GridTools
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.a = np.zeros (domain)
        self.d = np.zeros (domain)


    def stage_f0 (self, a, b, c):
        for p in self.get_interior_points (a):
            a[p] = b[p + (-1,0,0)] + b[p + (1,0,0)] + c[p + (1,0,0)]


    def stage_f1 (self, a, b, d):
        for p in self.get_interior_points (d):
            d[p] = b[p + (-2,0,0)] + a[p + (-1,0,0)] + a[p + (2,0,0)]


    def stage_f2 (self, a, c, d, e):
        for p in self.get_interior_points (a):
            e[p] = (a[p + (-1,0,0)] + a[p + (2,0,0)]
                    + d[p + (-2,0,0)] + d[p + (2,0,0)]
                    + c[p + (-1,0,0)] + c[p + (1,0,0)])


    @Stencil.kernel
    def kernel (self, in_b, in_c, out_e):
        self.stage_f0 (a=self.a,
                       b=in_b,
                       c=in_c)

        self.stage_f1 (a=self.a,
                       b=in_b,
                       d=self.d)

        self.stage_f2 (a=self.a,
                       c=in_c,
                       d=self.d,
                       e=out_e)



class GridToolsDataDeps2Test (CopyTest):
    """
    A test fixture for the GridToolsDataDeps2 stencil defined above.

    Contains a specific test case to validate individual access extents of
    stencil's input data fields.
    """
    __test__ = True


    def setUp (self):
        super ( ).setUp ( )
        logging.basicConfig (level=logging.INFO)
        self.domain = (16, 16, 8)
        self.params = ('in_b','in_c','out_e')
        self.temps  = ('self.a', 'self.d')

        self.in_b = np.ones (self.domain)
        self.in_c = np.ones (self.domain)
        self.out_e = np.zeros (self.domain)

        self.stencil = GridToolsDataDeps2 (self.domain)
        self.stencil.set_halo ( (4,5,0,0) )
        self.stencil.set_k_direction ('forward')


    def test_data_dependency_detection (self, expected_deps=None, backend='c++'):
        if expected_deps is None:
            expected_deps = [('out_e', 'in_c'),
                             ('out_e', 'self.d'),
                             ('out_e', 'self.a'),
                             ('self.d', 'in_b'),
                             ('self.d', 'self.a'),
                             ('self.a', 'in_b'),
                             ('self.a', 'in_c')]
        super ( ).test_data_dependency_detection (deps=expected_deps,
                                                  backend=backend)


    @attr(lang='cuda')
    def test_data_dependency_detection_cuda (self):
        self.test_data_dependency_detection (backend='cuda')


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS
        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_b', [-1,1,0,0])
        self.add_expected_offset ('in_b', [-2,0,0,0])
        self.add_expected_offset ('in_c', [0,1,0,0])
        self.add_expected_offset ('in_c', [-1,1,0,0])
        self.add_expected_offset ('out_e', None)
        self.add_expected_offset ('self.a', None)
        self.add_expected_offset ('self.a', [-1,2,0,0])
        self.add_expected_offset ('self.a', [-1,2,0,0])
        self.add_expected_offset ('self.d', None)
        self.add_expected_offset ('self.d', [-2,2,0,0])

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_ghost_cell_pattern (self, expected_patterns=None, backend='c++'):
        if expected_patterns is None:
            expected_patterns = {'stage_f0':[3,4,0,0],
                                 'stage_f1':[2,2,0,0],
                                 'stage_f2':[0,0,0,0] }
        super ( ).test_ghost_cell_pattern (expected_patterns,
                                           backend=backend)


    @attr(lang='cuda')
    def test_ghost_cell_pattern_cuda (self):
        self.test_ghost_cell_pattern (backend='cuda')


    def test_minimum_halo_detection (self, min_halo=None):
        if min_halo is None:
            min_halo = [4, 5, 0, 0]
        super ( ).test_minimum_halo_detection (min_halo)


    @unittest.skip("Not yet implemented")
    @attr(lang='python')
    def test_python_results (self):
        pass


    def test_parameter_access_extents (self):
        self._run ( )

        in_b = self.stencil.scope.symbol_table['in_b']
        in_c = self.stencil.scope.symbol_table['in_c']

        self.assertEqual (in_b.access_extent, [-4,5,0,0])
        self.assertEqual (in_c.access_extent, [-3,5,0,0])
