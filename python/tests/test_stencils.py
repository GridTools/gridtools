import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil import MultiStageStencil, StencilInspector




class RangeDetectionTest (unittest.TestCase):
    def setUp (self):
        logging.basicConfig (level=logging.INFO)
        self.field_ranges = dict ( )


    def add_expected_range (self, field, expected_range):
        if field not in self.field_ranges.keys ( ):
            self.field_ranges[field] = list ( )
        self.field_ranges[field].append (expected_range)


    def automatic_range_detection (self, stencil):
        import copy

        fld_rgs = copy.deepcopy (self.field_ranges)
        #
        # check the range detection within each functor of the stencil
        #
        for f in stencil.inspector.functors:
            sc = f.scope
            for p in sc.get_all ( ):
                if not sc.is_alias (p.name):
                    try:
                        expected = fld_rgs[p.name]
                        self.assertIn (sc[p.name].range, expected,
                                       "Range '%s' of '%s' does not match any of %s" %
                                       (sc[p.name].range, p.name, expected))
                        #
                        # remove the correct range to avoid finding it twice
                        #
                        index = fld_rgs[p.name].index (sc[p.name].range)
                        fld_rgs[p.name].pop (index)
                    except KeyError:
                        logging.error ("No range given for field '%s'"
                                       % p.name)
                        self.assertTrue (False)


    

class Copy (MultiStageStencil):
    """
    Definition of a simple copy stencil, as in 'examples/copy_stencil.h'.-
    """
    def kernel (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]



class CopyTest (RangeDetectionTest):
    """
    A test case for the copy stencil defined above.-
    """
    def _run (self):
        kwargs = dict ( )
        for p in self.params:
            kwargs[p] = getattr (self, p)
        self.stencil.run (**kwargs)


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

        self.stencil = Copy ( )
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_automatic_dependency_detection (self, deps=None, backend='c++'):
        self.stencil.backend = backend
        self._run ( )

        if deps is None:
            deps = [ self.params ]

        stencil_deps = self.stencil.scope.depency_graph.edges ( )
        #
        # check the dependency detection for the whole stencil
        #
        for d in deps:
            found = False
            for sd in stencil_deps:
                if d[0] == sd[0].name and d[1] == sd[1].name:
                    found = True
                    break
            if not found:
                logging.error ("Dependency %s not found in %s" % (sd, stencil_deps))
            self.assertTrue (found)


    def test_automatic_dependency_detection_cuda (self, deps=None, backend='cuda'):
        self.test_automatic_dependency_detection (deps=deps,
                                                  backend=backend)


    def test_automatic_range_detection (self):
        from gridtools import BACKENDS

        #
        # add the fields and its ranges
        #
        self.add_expected_range ('in_cpy',  None)
        self.add_expected_range ('out_cpy', None)

        for backend in BACKENDS:
            self.stencil.backend = backend
            self._run ( )
            self.automatic_range_detection (self.stencil)


    @attr(lang='cuda')
    def test_compare_python_cpp_and_cuda_results (self):
        import copy
        import random
        from gridtools import BACKENDS

        for backend in BACKENDS:
            stencil_native         = copy.deepcopy (self.stencil)
            stencil_native.backend = backend

            #
            # data fields - Py and C++ sets
            #
            params_py  = dict ( )
            params_cxx = dict ( )
            for p in self.params:
                params_py[p]  = np.random.rand (*self.domain)
                params_cxx[p] = np.copy (params_py[p])
            #
            # apply both stencils 10 times and compare the results
            # using an error threshold
            #
            err  = np.zeros (self.domain)
            err += 10e-12
            for i in range (10):
                self.stencil.run   (**params_py)
                stencil_native.run (**params_cxx)
                #
                # compare field contents
                #
                for k in params_py.keys ( ):
                    self.assertTrue (np.all (np.less (params_py[k] - params_cxx[k],
                                                      err)))


    def test_symbol_discovery (self, backend='c++'):
        self.stencil.backend = backend
        self._run ( )
        #
        # check fields were correctly discovered
        #
        scope = self.stencil.scope
        for p in self.params:
            self.assertTrue (scope.is_parameter (p))
        for t in self.temps:
            self.assertTrue (scope.is_temporary (t))


    def test_symbol_discovery_cuda (self):
        self.test_symbol_discovery (backend='cuda')


    def test_user_stencil_extends_multistagestencil (self):
        with self.assertRaises (TypeError):
            class DoesNotExtendAndShouldFail (object):
                pass
            insp = StencilInspector (DoesNotExtendAndShouldFail)

        insp = self.stencil.inspector
        insp.static_analysis ( )
        self.assertNotEqual (insp, None)


    def test_kernel_function (self):
        """
        The kernel function is the entry point of the stencil execution and
        should follow several conventions.-
        """
        #
        # FIXME will not work because the 'class' definition is indented and
        #       it should not be
        #
        """
        with self.assertRaises (NameError):
            class KernelFunctionMissing (MultiStageStencil):
                def some_func (self):
                    return None
            insp = StencilInspector (KernelFunctionMissing)
            insp.analyze ( )
        with self.assertRaises (ValueError):
            class KernelFunctionShouldReturnNone (MultiStageStencil):
                def kernel (self):
                    return "something"
            insp = StencilInspector (KernelFunctionDoesNotReturnNone)
            insp.analyze ( )
        """
        pass


    def test_run_stencil_only_accepts_keyword_arguments (self):
        with self.assertRaises (KeyError):
            self.stencil.run ([ getattr (self, p) for p in self.params ])


    @attr(lang='python')
    def test_python_results (self, out_param=None, result_file=None):
        import os

        cur_dir = os.path.dirname (os.path.abspath (__file__))

        self.stencil.backend = 'python'
        self._run ( )
        #
        # take halo into account when comparing the results
        #
        beg_i = self.stencil.halo[0]
        end_i = self.domain[0] - self.stencil.halo[1]
        beg_j = self.stencil.halo[2]
        end_j = self.domain[1] - self.stencil.halo[3]

        if result_file is None:
            expected  = self.in_cpy
            out_param = 'out_cpy'
        else:
            expected = np.load ('%s/%s' % (cur_dir,
                                           result_file)) 
        self.assertTrue (np.array_equal (getattr (self, out_param)[beg_i:end_i, beg_j:end_j],
                                         expected[beg_i:end_i, beg_j:end_j]))


    def test_execution_performance_cpp (self, backend='c++'):
        import time

        self.stencil.backend = backend
        self._run ( )

        nstep  = 100
        tstart = time.time ( )
        for i in range (nstep):
            self._run ( )
        print ('FPS:', nstep / (time.time()-tstart))


    def test_execution_performance_cuda (self):
        self.test_execution_performance_cpp (backend='cuda')


    def test_k_directions (self, backend='c++'):
        self.stencil.backend = backend
        for dir in ('forward', 'backward'):
            self.stencil.set_k_direction (dir)
            self._run ( )


    def test_k_directions_cuda (self):
        self.test_k_directions (backend='cuda')



class FloatPrecisionTest (CopyTest):
    """
    Tests for the exceptions raised by the floating point precision validation.
    """
    def test_float_input_type_validation_wrong_data_size (self):
        with self.assertRaises (TypeError):
            self.stencil.backend = 'c++'
            self.in_cpy = self.in_cpy.astype(np.float16)
            self._run ( )


    def test_float_input_type_validation_wrong_data_type (self):
        with self.assertRaises (TypeError):
            self.stencil.backend = 'c++'
            self.in_cpy = self.in_cpy.astype(np.int64)
            self._run ( )


    def test_float_input_type_validation_potentially_correct_type_but_not_the_correct_size (self):
        with self.assertRaises (TypeError):
            self.stencil.backend = 'c++'
            self.in_cpy = self.in_cpy.astype(np.float32)
            self._run ( )




class Power (MultiStageStencil):
    """
    Immitates the CopyStencil using the power operator.-
    """
    def kernel (self, out_cpy, in_cpy):
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
            out_cpy[p] =  5 **  0
            out_cpy[p] =  5 **  1
            out_cpy[p] =  5 **  2
            out_cpy[p] =  5 **  3
            out_cpy[p] =  5 **  +3
            out_cpy[p] =  (-5) **  +3
            out_cpy[p] =  5 ** -1
            out_cpy[p] =  5 ** -2
            out_cpy[p] =  5 ** -2*3
            out_cpy[p] =  5 ** (-2*3)
            out_cpy[p] = in_cpy[p] ** -1
            out_cpy[p] = -in_cpy[p] ** -1
            out_cpy[p] = (-in_cpy[p]) ** -1
            out_cpy[p] = (-in_cpy[p]) ** -3
            out_cpy[p] = (in_cpy[p] ** -1) * (in_cpy[p] ** 2)
            out_cpy[p] = (((-in_cpy[p]) ** -2) * ((-in_cpy[p]) ** 2)) * (in_cpy[p])
            # The final statement so that we can rerun test
            out_cpy[p] = (in_cpy[p]**2)*(in_cpy[p] ** -1)



class PowerTest (CopyTest):
    def setUp (self):
        super ( ).setUp ( )
        self.stencil = Power ( )




class Laplace (MultiStageStencil):
    """
    A Laplacian operator, as the one used in COSMO.-
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        #
        # temporary data fields to share data among the different stages
        #
        self.out_cpy = np.zeros (domain)


    def kernel (self, out_data, in_data):
        """
        Stencil's entry point.-
        """
        for p in self.get_interior_points (out_data):
            self.out_cpy[p] = -4.0 * in_data[p] + (
                              in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                              in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )

        for p in self.get_interior_points (out_data):
            out_data[p] = self.out_cpy[p]



class LaplaceTest (CopyTest):
    """
    Testing the Laplace operator.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.domain = (64, 64, 32)
        self.params = ('out_data', 'in_data')
        self.temps  = ( )

        self.out_data = np.zeros (self.domain)
        self.in_data  = np.zeros (self.domain)
        for i in range (self.domain[0]):
            for j in range (self.domain[1]):
                for k in range (self.domain[2]):
                    self.in_data[i,j,k] = i**3 + j

        self.stencil = Laplace (self.domain) 
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_automatic_range_detection (self):
        from gridtools import BACKENDS

        #
        # add the fields and its ranges
        #
        self.add_expected_range ('in_data', [-1,1,-1,1])
        self.add_expected_range ('out_cpy', None)

        for backend in BACKENDS:
            self.stencil.backend = backend
            self._run ( )
            self.automatic_range_detection (self.stencil)


    @attr(lang='python')
    def test_python_results (self):
        self.out_data = np.random.rand (*self.domain)
        super ( ).test_python_results (out_param='out_data',
                                       result_file='laplace_result.npy')




class HorizontalDiffusion (MultiStageStencil):
    def __init__ (self, domain):
        super ( ).__init__ ( )
        #
        # temporary data fields to share data among the different stages
        #
        self.lap = np.zeros (domain)
        self.fli = np.zeros (domain)
        self.flj = np.zeros (domain)


    def stage_flux_i (self, out_fli, in_lap):
        for p in self.get_interior_points (in_lap):
            out_fli[p] = in_lap[p + (1,0,0)] - in_lap[p]


    def stage_flux_j (self, out_flj, in_lap):
        for p in self.get_interior_points (in_lap):
            out_flj[p] = in_lap[p + (0,1,0)] - in_lap[p]


    def kernel (self, out_data, in_data, in_wgt):
        #
        # Laplace
        #
        for p in self.get_interior_points (in_data):
            self.lap[p] = -4.0 * in_data[p] +  (
                          in_data[p + (-1,0,0)] + in_data[p + (1,0,0)] +
                          in_data[p + (0,-1,0)] + in_data[p + (0,1,0)] )
        #
        # The fluxes over 'i' and 'j' are independent
        #
        self.stage_flux_i (out_fli = self.fli,
                           in_lap  = self.lap)
        self.stage_flux_j (out_flj = self.flj,
                           in_lap  = self.lap)
        #
        # Last stage
        #
        for p in self.get_interior_points (out_data):
            out_data[p] = in_wgt[p] * (
                          self.fli[p + (-1,0,0)] - self.fli[p] + 
                          self.flj[p + (0,-1,0)] - self.flj[p] )



class HorizontalDiffusionTest (CopyTest):
    """
    A test case for the HorizontalDiffusion stencil defined above.-
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

        self.stencil = HorizontalDiffusion (self.domain)
        self.stencil.set_halo ( (2, 2, 2, 2) )
        self.stencil.set_k_direction ("forward")


    def test_automatic_dependency_detection (self, deps=None, backend='c++'):
        expected_deps = [('out_data', 'in_wgt'),
                         ('out_data', 'self.flj'),
                         ('out_data', 'self.fli'),
                         ('self.fli', 'self.lap'),
                         ('self.flj', 'self.lap'),
                         ('self.lap', 'in_data')]
        super ( ).test_automatic_dependency_detection (deps=expected_deps)
        super ( ).test_automatic_dependency_detection (deps=expected_deps,
                                                       backend='cuda')


    def test_automatic_range_detection (self):
        from gridtools import BACKENDS

        #
        # add the fields and its ranges
        #
        self.add_expected_range ('in_data',  [-1,1,-1,1])
        self.add_expected_range ('in_wgt',   None)
        self.add_expected_range ('out_data', None)
        self.add_expected_range ('self.fli', None)
        self.add_expected_range ('self.fli', [-1,0,0,0])
        self.add_expected_range ('self.flj', None)
        self.add_expected_range ('self.flj', [0,0,-1,0])
        self.add_expected_range ('self.lap', None)
        self.add_expected_range ('self.lap', [0,1,0,0])
        self.add_expected_range ('self.lap', [0,0,0,1])

        for backend in BACKENDS:
            self.stencil.backend = backend
            self._run ( )
            self.automatic_range_detection (self.stencil)


    @attr(lang='python')
    def test_python_results (self):
        self.out_data = np.random.rand (*self.domain)
        super ( ).test_python_results (out_param='out_data',
                                       result_file='horizontaldiffusion_result.npy')



class ChildStencilCallsParentConstructorAndNothingElse (MultiStageStencil):
    """
    Child constructor correctly calls parent constructor and has no other work, comments or docstrings.
    Works correctly--no exceptions.
    """
    def __init__ (self): super ( ).__init__ ( )


    def kernel (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]



class ChildStencilCallsParentConstructorFirst (MultiStageStencil):
    """
    Child constructor correctly calls parent constructor before doing any other work.
    This test shows that it does not matter what arbitrary work is performed after
    the call to the parent constructor.  If the call to the parent constructor is
    first, it works.
    Works correctly--no exceptions.
    """
    def __init__ (self):
        super ( ).__init__ ( )
        """ This is a Python feature called a docstring.  """
        """ Here is another docstring """
        anum = 2
        bnum = anum * 4
        cstr = "string" + " operation"
        dstr = "This RHS is a string."
        enum = 5
        # This line is a comment.
        fnum = 5
        """
        These lines resemble a comment but are actually
        a single string.
        """
        gnum = 22


    def kernel (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]



class ChildStencilCallsParentConstructorAfterComment (MultiStageStencil):
    """
    Child constructor correctly calls parent constructor with only a comment coming before it.
    Works correctly--no exceptions.
    """
    def __init__ (self):
        # This is a comment.
        super ( ).__init__ ( )


    def kernel (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]



class ChildStencilCallsParentConstructorAfterMultComments (MultiStageStencil):
    """
    Child constructor calls parent constructor before doing any other work.
    There are only comments preceding it.
    Works correctly--no exceptions.
    """
    def __init__ (self):
        # This is a comment.
        # This is another comment.
        super ( ).__init__ ( )


    def kernel (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]



class ChildStencilCallsParentConstructorAfterDocString (MultiStageStencil):
    """
    Child constructor correctly calls parent constructor before doing any other work.
    There is only a single string, which is treated by Python as a docstring, that
    precedes the call.
    Works correctly--no exceptions.
    """
    def __init__ (self):
        """ This is a Python feature called a docstring.  """
        super ( ).__init__ ( )


    def kernel (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]



class ChildStencilCallsParentConstructorAfterMultDocStrings (MultiStageStencil):
    """
    Child constructor calls parent constructor before doing any other work
    but there is more than one docstring and this causes the exception to be raised.
    Should raise a ReferenceError.
    """
    def __init__ (self):
        """ This is a Python feature called a docstring.  """
        """ This is another docstring.  """
        super ( ).__init__ ( )



class ChildStencilNoCallParentConstructor (MultiStageStencil):
    """
    Child constructor does not call the parent constructor at all.
    Should raise a ReferenceError.
    """
    def __init__ (self):
        """ This is a Python feature called a docstring.  """
        """ Here is another docstring """
        # Since this class tests the case where we don't call the parent constructor,
        # a side effect is that there is no inspector field which would've been
        # defined by the grandparent constructor (that of the Stencil class).  Since
        # we do NOT wish to test the AttributeError here but rather the ReferenceError,
        # the same declaration of inspector that exists in Stencil has been placed here.
        self.inspector = StencilInspector (self)



class ChildStencilCallsParentConstructorAfterAssignment (MultiStageStencil):
    """
    Child constructor performs a numerical assignment before calling the parent constructor.
    Should raise a ReferenceError.
    """
    def __init__ (self):
        anum = 1
        super ( ).__init__ ( )



class ChildStencilCallsParentConstructorAfterStringAssignment (MultiStageStencil):
    """
    Child constructor performs a string assignment before calling the parent constructor.
    Should raise a ReferenceError.
    """
    def __init__ (self):
        astr = "String assignment"
        super ( ).__init__ ( )



class ChildStencilParentConstructorAfterComputation (MultiStageStencil):
    """
    Child constructor performs a computation before calling parent constructor.
    Should raise a ReferenceError.
    """
    def __init__ (self):
        2 * 3
        super ( ).__init__ ( )
        # This line is a comment.



class ChildStencilTest (unittest.TestCase):
    """
    A test case for the copy stencil defined above.-
    """
    def _run (self):
        kwargs = dict ( )
        for p in self.params:
            kwargs[p] = getattr (self, p)
        self.stencil.run (**kwargs)


    def setUp (self):
        logging.basicConfig (level=logging.INFO)

        self.stencil = None


    def test_child_constructor_calls_parent_constructor_and_nothing_else (self):
        self.stencil = ChildStencilCallsParentConstructorAndNothingElse ( )
        insp = self.stencil.inspector
        insp.static_analysis ( )
        self.assertNotEqual (insp, None)


    def test_child_constructor_calls_parent_constructor_first (self):
        self.stencil = ChildStencilCallsParentConstructorFirst ( )
        insp = self.stencil.inspector
        insp.static_analysis ( )
        self.assertNotEqual (insp, None)


    def test_child_constructor_calls_parent_constructor_after_comment (self):
        self.stencil = ChildStencilCallsParentConstructorAfterComment ( )
        insp = self.stencil.inspector
        insp.static_analysis ( )
        self.assertNotEqual (insp, None)


    def test_child_constructor_calls_parent_constructor_after_mult_comments (self):
        self.stencil = ChildStencilCallsParentConstructorAfterMultComments ( )
        insp = self.stencil.inspector
        insp.static_analysis ( )
        self.assertNotEqual (insp, None)


    def test_child_constructor_calls_parent_constructor_after_doc_string (self):
        self.stencil = ChildStencilCallsParentConstructorAfterDocString ( )
        insp = self.stencil.inspector
        insp.static_analysis ( )
        self.assertNotEqual (insp, None)


    def test_child_constructor_calls_parent_constructor_after_mult_doc_strings (self):
        with self.assertRaises (ReferenceError):
            self.stencil = ChildStencilCallsParentConstructorAfterMultDocStrings ( )
            insp = self.stencil.inspector
            insp.static_analysis ( )


    def test_child_constructor_no_call_parent_constructor (self):
        with self.assertRaises (ReferenceError):
            self.stencil = ChildStencilNoCallParentConstructor ( )
            insp = self.stencil.inspector
            insp.static_analysis ( )


    def test_child_constructor_calls_parent_constructor_after_assignment (self):
        with self.assertRaises (ReferenceError):
            self.stencil = ChildStencilCallsParentConstructorAfterAssignment ( )
            insp = self.stencil.inspector
            insp.static_analysis ( )


    def test_child_constructor_calls_parent_constructor_after_string_assignment (self):
        with self.assertRaises (ReferenceError):
            self.stencil = ChildStencilCallsParentConstructorAfterStringAssignment ( )
            insp = self.stencil.inspector
            insp.static_analysis ( )


    def test_child_constructor_calls_parent_constructor_after_computation (self):
        with self.assertRaises (ReferenceError):
            self.stencil = ChildStencilParentConstructorAfterComputation ( )
            insp = self.stencil.inspector
            insp.static_analysis ( )
