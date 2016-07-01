# -*- coding: utf-8 -*-
import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil import Stencil, MultiStageStencil



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
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
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



class ReturnSomethingKernel (MultiStageStencil):
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
    def kernel (self, out):
        """
        A kernel function should return None. Here we return a different object
        Also calling another function that returns a value for further testing.
        """
        #
        # iterate over the points
        #
        for p in self.get_interior_points (out):
              out[p] = out[p]*self.return2 ( )

        return 'something'


    def return2(self):
        return 2



class ReturnSomethingKernelTest (EmptyKernelTest):
    """
    A test case for the ReturnSomethingKernel stencil defined above.
    """
    def setUp (self):
        super ( ).setUp ( )

        self.domain = (64, 64, 32)
        self.params = ('out',)

        self.out = np.ones (self.domain,
                            dtype=np.float64,
                            order='F')

        self.stencil = ReturnSomethingKernel ( )
        self.error = ValueError



class MultipleKernels (MultiStageStencil):
    """
    Definition of a stencil with multiple kernels
    """
    def __init__ (self):
        super ( ).__init__ ( )

    @Stencil.kernel
    def kernel1 (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]


    @Stencil.kernel
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
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @Stencil.kernel
    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] is out_X[p]:
                out_X[p] = 0.0



class IfStatementOpIsNotFailure (MultiStageStencil):
    """
    Tests that use of 'is not' operator currently raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @Stencil.kernel
    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] is not out_X[p]:
                out_X[p] = 0.0



class IfStatementOpNotInFailure (MultiStageStencil):
    """
    Tests that use of 'not in' operator currently raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @Stencil.kernel
    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] not in out_X[p]:
                out_X[p] = 0.0



class IfStatementOpInFailure (MultiStageStencil):
    """
    Tests that use of 'in' operator currently raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @Stencil.kernel
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

        self.stencil = IfStatementOpIsFailure ( )
        self.error = NotImplementedError



class IfStatementsOpIsNotTest (IfStatementsOpIsTest):
    """
    A test case for the 'If + is not' statement related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = IfStatementOpIsNotFailure ( )



class IfStatementsOpNotInTest (IfStatementsOpIsTest):
    """
    A test case for the 'If + not in' statement related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = IfStatementOpNotInFailure ( )



class IfStatementsOpInTest (IfStatementsOpIsTest):
    """
    A test case for the 'If + in' statement related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = IfStatementOpInFailure ( )



class SelfDependHD (MultiStageStencil):
    """
    A stencil featuring data self-dependency, derived from HorizontalDiffusion
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        #
        # temporary data fields to share data among the different stages
        #
        self.lap = np.zeros (domain)
        self.fli = np.zeros (domain)
        self.flj = np.zeros (domain)


    def stage_laplace (self, out_lap, in_data):
        for p in self.get_interior_points (out_lap,
                                          ghost_cell=[-1,1,-1,1]):
           out_lap[p] = -4.0 * in_data[p] +  (
                         in_data[p + (-1,0,0)] + in_data[p + (1,0,0)] +
                         in_data[p + (0,-1,0)] + in_data[p + (0,1,0)] )


    def stage_flux_i (self, out_fli, in_data):
        for p in self.get_interior_points (out_fli,
                                           ghost_cell=[-1,0,-1,0]):
            out_fli[p] = in_data[p + (1,0,0)] - in_data[p]


    def stage_flux_j (self, out_flj, in_data):
        for p in self.get_interior_points (out_flj,
                                           ghost_cell=[-1,0,-1,0]):
            out_flj[p] = in_data[p + (0,1,0)] - in_data[p]


    @Stencil.kernel
    def kernel (self, out_data, in_data, in_wgt):
        #
        # Laplace
        #
        self.stage_laplace (out_lap=self.lap,
                           in_data=in_data)
        #
        # the fluxes are independent, because they depend on 'self.lap'
        #
        self.stage_flux_i (out_fli = self.fli,
                           in_data  = self.lap)
        self.stage_flux_j (out_flj = self.flj,
                           in_data  = self.lap)

        for p in self.get_interior_points (self.fli,ghost_cell=[-1,0,-1,0]):
           #
           # Data field self-assignment
           # fli = fli + flj
           #
           self.fli[p] = (self.fli[p + (-1,0,0)] - self.fli[p] +
                          self.flj[p + (0,-1,0)] - self.flj[p] )

        for p in self.get_interior_points (out_data):
            #
            # Last stage
            #
            out_data[p] = in_wgt[p] * (
                          self.fli[p + (-1,0,0)] - self.fli[p] +
                          self.flj[p + (0,-1,0)] - self.flj[p] )



class SelfDependHDTest (IfStatementsOpIsTest):
    """
    A test case for the SelfDependHD stencil defined above, reusing the
    HorizontalDiffusion test case.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.domain = (64, 64, 32)
        self.params = ('out_data',
                       'in_data',
                       'in_wgt')
        self.temps  = ('self.lap',
                       'self.fli',
                       'self.flj')

        self.out_data = np.zeros (self.domain)
        self.in_wgt   = np.ones  (self.domain)
        self.in_data  = np.zeros (self.domain)

        for i in range (self.domain[0]):
            for j in range (self.domain[1]):
                for k in range (self.domain[2]):
                    self.in_data[i,j,k] = i**5 + j

        self.stencil = SelfDependHD (self.domain)
        self.stencil.set_halo ( (2, 2, 2, 2) )
        self.stencil.set_k_direction ("forward")
        self.error = ValueError


class VRForLoopStage (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing inside a for-loop defined
    stage raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    @Stencil.kernel
    def kernel (self, out_X):
        for p in self.get_interior_points (out_X[:,:,10:25]):
            out_X[p] = 1.0



class VRForLoopStageTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil
    defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRForLoopStage ( )
        self.error = RuntimeError



class VR2DSlice (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing on less than 3 dimensions
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[:,:]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VR2DSliceTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VR2DSlice ( )
        self.error = ValueError



class VRSingleIndex (MultiStageStencil):
    """
    Tests that using get_interior_points() with single-index slicing
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[:,:,10]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRSingleIndexTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRSingleIndex ( )
        self.error = ValueError



class VRISlice (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing on i direction
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[5:,:,:]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRISliceTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRISlice ( )
        self.error = ValueError



class VRJSlice (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing on j direction
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[:,:17,:]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRJSliceTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRJSlice ( )
        self.error = ValueError



class VRNegativeSliceIndex (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing on k direction and a
    negative index raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[:,:,:-2]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRNegativeSliceIndexTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRNegativeSliceIndex ( )
        self.error = NotImplementedError



class VRSliceStepsK (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing steps
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[:,:,8:16:2]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRSliceStepsKTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRSliceStepsK ( )
        self.error = ValueError



class VRSliceStepsJ (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing steps
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[:,::2,:]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRSliceStepsJTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRSliceStepsJ ( )
        self.error = ValueError



class VRSliceStartOverflow (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing index overflow at start
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[:,:,35:]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRSliceStartOverflowTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRSliceStartOverflow ( )
        self.error = ValueError



class VRSliceEndOverflow (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing index overflow at end
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        for p in self.get_interior_points (out_X[:,:,12:36]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRSliceEndOverflowTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRSliceEndOverflow ( )
        self.error = ValueError



class VRNonNumSliceIndex (MultiStageStencil):
    """
    Tests that using get_interior_points() with slicing and non-numerical index
    raises an error.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def stage_1 (self, out_X):
        top = 30
        for p in self.get_interior_points (out_X[:,:,12:top]):
            out_X[p] = 1.0


    @Stencil.kernel
    def kernel (self, out_X):
        self.stage_1 (out_X = out_X)



class VRNonNumSliceIndexTest (IfStatementsOpIsTest):
    """
    A test case for the Vertical Regions related stencil defined above.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = VRNonNumSliceIndex ( )
        self.error = NotImplementedError