# -*- coding: utf-8 -*-
import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil  import MultiStageStencil, def_kernel



class NoKernelTest (unittest.TestCase):
    """
    Tests that stencils with no kernels correctly raise errors and are unregistered
    """
    def _run (self, stencil):
        kwargs = dict ( )
        for p in self.params:
            kwargs[p] = getattr (self, p)
        stencil.run (**kwargs)


    def setUp (self):
        super ( ).setUp ( )
        logging.basicConfig (level=logging.INFO)

        self.params = ()

        self.stencil = MultiStageStencil ()
        self.error = AttributeError


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



class EmptyKernelTest (NoKernelTest):
    """
    A base test case for stencils with invalid kernels.
    """
    def setUp (self):
        super ( ).setUp ( )

        self.params = ('out_arg', 'in_arg')

        self.out_arg = None
        self.in_arg = None

        self.stencil = EmptyKernel ( )
        self.error = NameError



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



class MultipleKernelsTest (NoKernelTest):
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



class IfStatementsOpIsTest (NoKernelTest):
    """
    A test case for the 'If + is' statement related stencil defined above.-
    May serve as a base test class for stencils with unsupported language
    features in the kernel
    """


    def setUp (self):
        super ( ).setUp ( )
        self.domain = (64, 64, 32)
        self.params = ('out_X','in_X')

        self.in_X = np.random.random_integers (10, size=self.domain)
        self.in_X = self.in_X.astype (np.float64)

        self.out_X = np.copy (self.in_X)

        self.stencil = IfStatementOpIsFailure (self.domain)
        self.error = NotImplementedError



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
