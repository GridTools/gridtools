import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil  import MultiStageStencil, stencil_kernel
from tests.test_stencils import CopyTest



class GameOfLife (MultiStageStencil):
    """
    # Tests various parts of the if-statement, notably:
    # 1) Single-clause conditional
    # 2) Multi-clause conditional connected by "and"
    # 3) Single-statement body
    # 4) Single-statement else
    # 5) Else-if block
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.counter = np.zeros (domain)


    @stencil_kernel
    def kernel (self, out_X, in_X):
        for p in self.get_interior_points (out_X):

            self.counter[p] = out_X[p + (1,0,0)]  + out_X[p + (1,1,0)]   + \
                              out_X[p + (0,1,0)]  + out_X[p + (-1,1,0)]  + \
                              out_X[p + (-1,0,0)] + out_X[p + (-1,-1,0)] + \
                              out_X[p + (0,-1,0)] + out_X[p + (1,-1,0)]

            if out_X[p] == 1.0 and self.counter[p] == 2:
                out_X[p] = 0.0
            elif self.counter[p] == 3:
                out_X[p] = 1.0
            else:
                out_X[p] = 0.0



class GameOfLifeTest (CopyTest):
    """
    A test case for the GameOfLife stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )
        logging.basicConfig (level=logging.INFO)
        self.domain = (64, 64, 32)
        self.params = ('out_X','in_X')

        self.in_X = np.random.random_integers (10,
                                               size=self.domain)
        self.in_X = self.in_X.astype (np.float64)

        self.out_X = np.copy (self.in_X)

        self.stencil = GameOfLife (self.domain)
        self.stencil.set_halo ( (1,1,1,1) )


    @attr(lang='cuda')
    def test_compare_python_cpp_and_cuda_results (self):
        super ( ).test_compare_python_cpp_and_cuda_results ( )


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS

        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_X',  None)
        self.add_expected_offset ('out_X', [-1,1,-1,1])
        self.add_expected_offset ('self.counter', None)

        for backend in BACKENDS:
            self.stencil.backend = backend
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_data_dependency_detection (self, deps=None, backend='c++'):
        expected_deps = [('self.counter','out_X')]
        super ( ).test_data_dependency_detection (deps=expected_deps,
                                                  backend=backend)


    @attr(lang='cuda')
    def test_data_dependency_detection_cuda (self):
        self.test_data_dependency_detection (backend='cuda')


    def test_minimum_halo_detection (self):
        super ( ).test_minimum_halo_detection ([1, 1, 1, 1])

    @unittest.skip("Not yet implemented")
    @attr(lang='python')
    def test_python_results (self):
        pass



class AdditionalIfStatement (MultiStageStencil):
    """
    # Additional tests of the if-statement, notably:
    # 1) Multi-clause conditional connected by "and" and "or"
    # 2) Multi-clause conditional connected by "and" and "or" with parenthesis
    # 3) Multi-statement body
    # 4) Multi-statement else
    # 5) Not operator used within the conditional
    # 6) Uses the following relational operators: >, <, <=, >=, ==, !=
    # 7) Tests an if-statment without any else clause
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.counter = np.zeros (domain)


    @stencil_kernel
    def kernel (self, out_X, in_X):
        for p in self.get_interior_points (out_X):
            self.counter[p] = out_X[p + (1,0,0)]

            if (out_X[p] != 1.0 and self.counter[p] == 2 and out_X[p + (1,0,0)] > 0.0) or self.counter[p] < 10:
                out_X[p] = 0.0
            if out_X[p] == 2.0 and (self.counter[p] <= 10 or out_X[p] < 3.2):
                out_X[p] = 1.0

            if self.counter[p] == 1.0:
                out_X[p] = 0.0
            elif self.counter[p] >= 3:
                out_X[p] = 1.0
                out_X[p + (1,0,0)] = 1.0
            else:
                out_X[p] = 0.0
                out_X[p] = 1.0
                out_X[p] = 2.0

            if not out_X[p]:
                out_X[p] = 0.5



class AdditionalIfStatementTest (CopyTest):
    """
    A test case for the AdditionalIfStatement stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )
        logging.basicConfig (level=logging.INFO)
        self.domain = (64, 64, 32)
        self.params = ('out_X','in_X')

        self.in_X = np.random.random_integers (10,
                                               size=self.domain)
        self.in_X = self.in_X.astype (np.float64)

        self.out_X = np.copy (self.in_X)

        self.stencil = AdditionalIfStatement (self.domain)
        self.stencil.set_halo ( (0,1,0,0) )


    @attr(lang='cuda')
    def test_compare_python_cpp_and_cuda_results (self):
        super ( ).test_compare_python_cpp_and_cuda_results ( )


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS

        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_X',  None)
        self.add_expected_offset ('out_X', [0,1,0,0])
        self.add_expected_offset ('self.counter', None)

        for backend in BACKENDS:
            self.stencil.backend = backend
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_data_dependency_detection (self, deps=None, backend='c++'):
        expected_deps = [('self.counter', 'out_X')]
        super ( ).test_data_dependency_detection (deps=expected_deps,
                                                  backend=backend)


    @attr(lang='cuda')
    def test_data_dependency_detection_cuda (self):
        self.test_data_dependency_detection (backend='cuda')


    def test_minimum_halo_detection (self):
        super ( ).test_minimum_halo_detection ([0, 1, 0, 0])


    @unittest.skip("Not yet implemented")
    @attr(lang='python')
    def test_python_results (self):
        pass