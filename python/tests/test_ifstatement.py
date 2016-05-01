import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil  import MultiStageStencil, def_kernel
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


    @def_kernel
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


    @def_kernel
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



class EmptyKernel (MultiStageStencil):
    """
    Definition of a simple stencil with invalid kernel
    """
    @def_kernel
    def kernel (self, out_arg, in_arg):
        """
        Just an empty kernel
        """
        pass



class EmptyKernelTest (unittest.TestCase):
    """
    A base test case for stencils with invalid kernels.
    """
    def _run (self, stencil):
        kwargs = dict ( )
        for p in self.params:
            kwargs[p] = getattr (self, p)
        stencil.run (**kwargs)


    def setUp (self):
        super ( ).setUp ( )
        logging.basicConfig (level=logging.INFO)

        self.params = ('out_arg', 'in_arg')

        self.out_arg = None
        self.in_arg = None

        self.stencil = EmptyKernel ( )
        self.error = NameError


    def test_raises_error (self, backend='c++'):
        self.stencil.backend = backend
        with self.assertRaises (self.error):
            self._run(self.stencil)


    @attr(lang='cuda')
    def test_raises_error_cuda (self):
        self.test_raises_error(backend='cuda')


    @attr(lang='python')
    def test_raises_error_python (self):
        self.test_raises_error(backend='python')


    def test_unregister (self, backend='c++'):
        self.stencil.backend = backend
        with self.assertRaises (self.error):
            self._run(self.stencil)
        self.assertIs ( (self.stencil in self.stencil.compiler), False)


    @attr(lang='cuda')
    def test_unregister_cuda (self):
        self.test_unregister(backend='cuda')


    @attr(lang='python')
    def test_unregister_python (self):
        self.test_unregister(backend='python')



class NoKernelTest (EmptyKernelTest):
    """
    Tests that stencils with no kernels correctly raise errors and are unregistered
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = MultiStageStencil ()
        self.error = AttributeError



class MultipleKernels (MultiStageStencil):
    """
    Definition of a stencil with multiple kernels
    """
    @def_kernel
    def kernel1 (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]


    @def_kernel
    def kernel2 (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = 2*in_cpy[p]



class MultipleKernelsTest (EmptyKernelTest):
    """
    Tests that stencils with multiple kernels correctly raise errors and are unregistered
    The setUp replicates the one from CopyTest to test in a valid situation, if
    not for the kernels in excess
    """
    def setUp (self):
        super ( ).setUp ( )

        self.domain = (64, 64, 32)
        self.params = ('out_cpy', 'in_cpy')
        self.temps  = ( )

        self.out_cpy = np.zeros (self.domain,
                                 dtype=np.float64,
                                 order='F')
        #
        # workaround because of a bug in the power (**) implemention of NumPy
        #
        self.in_cpy = np.random.random_integers (10,
                                                 size=self.domain)
        self.in_cpy = self.in_cpy.astype (np.float64)
        self.in_cpy = np.asfortranarray (self.in_cpy)

        self.stencil = MultipleKernels ()
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")
        self.error = AttributeError



class IfStatementOpIsFailure (MultiStageStencil):
    """
    Tests that use of 'is' operator currently raises an error.
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @def_kernel
    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] is out_X[p]:
                out_X[p] = 0.0



class IfStatementOpIsNotFailure (MultiStageStencil):
    """
    Tests that use of 'is not' operator currently raises an error.
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @def_kernel
    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] is not out_X[p]:
                out_X[p] = 0.0



class IfStatementOpNotInFailure (MultiStageStencil):
    """
    Tests that use of 'not in' operator currently raises an error.
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @def_kernel
    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] not in out_X[p]:
                out_X[p] = 0.0



class IfStatementOpInFailure (MultiStageStencil):
    """
    Tests that use of 'in' operator currently raises an error.
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @def_kernel
    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] in out_X[p]:
                out_X[p] = 0.0



class IfStatementsOpIsTest (EmptyKernelTest):
    """
    A test case for the 'If + is' statement related stencil defined above.-
    May serve as a base test class for stencils with unsupported language
    features in the kernel
    """
    def _call_kernel (self):
        kwargs = dict ( )
        for p in self.params:
            kwargs[p] = getattr (self, p)
        self.stencil._kernel (**kwargs)


    def setUp (self):
        super ( ).setUp ( )
        self.domain = (64, 64, 32)
        self.params = ('out_X','in_X')

        self.in_X = np.random.random_integers (10, size=self.domain)
        self.in_X = self.in_X.astype (np.float64)

        self.out_X = np.copy (self.in_X)

        self.stencil = IfStatementOpIsFailure (self.domain)
        self.error = NotImplementedError


    def test_internal_kernel_reset (self, backend='c++'):
        self.stencil.backend = backend
        with self.assertRaises (self.error):
            self._run(self.stencil)
        with self.assertRaises (NotImplementedError):
            self._call_kernel ( )



class IfStatementsOpIsNotTest (IfStatementsOpIsTest):
    """
    A test case for the 'If + is not' statement related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = IfStatementOpIsNotFailure (self.domain)



class IfStatementsOpNotInTest (IfStatementsOpIsTest):
    """
    A test case for the 'If + not in' statement related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = IfStatementOpNotInFailure (self.domain)



class IfStatementsOpInTest (IfStatementsOpIsTest):
    """
    A test case for the 'If + in' statement related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = IfStatementOpInFailure (self.domain)
