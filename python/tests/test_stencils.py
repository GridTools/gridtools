import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil import Stencil, MultiStageStencil



class AccessPatternDetectionTest (unittest.TestCase):
    def setUp (self):
        logging.basicConfig (level=logging.INFO)
        self.field_access_patterns = dict ( )


    def add_expected_offset (self, field, expected_offset):
        if field not in self.field_access_patterns.keys ( ):
            self.field_access_patterns[field] = list ( )
        self.field_access_patterns[field].append (expected_offset)


    def automatic_access_pattern_detection (self, stencil):
        import copy

        acc_ptn = copy.deepcopy (self.field_access_patterns)
        #
        # check the access-pattern detection within each stencil stage
        #
        for stg in stencil.stages:
            sc = stg.scope
            for p in sc.get_all ( ):
                if not sc.is_alias (p.name):
                    try:
                        expected = acc_ptn[p.name]
                        self.assertIn (sc[p.name].access_pattern,
                                       expected,
                                       "Access offset '%s' of field '%s' does not match any of %s" %
                                        (sc[p.name].access_pattern,
                                         p.name,
                                         expected))
                        #
                        # remove the correct pattern to avoid finding it twice
                        #
                        index = acc_ptn[p.name].index (sc[p.name].access_pattern)
                        acc_ptn[p.name].pop (index)
                    except KeyError:
                        logging.error ("No access offsets given for field '%s'"
                                       % p.name)
                        self.assertTrue (False)



class Copy (MultiStageStencil):
    """
    Definition of a simple copy stencil, as in 'examples/copy_stencil.h'.-
    """
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
    def kernel (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
            out_cpy[p] = in_cpy[p]



class CopyTest (AccessPatternDetectionTest):
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


    def test_data_dependency_detection (self, deps=None, backend='c++'):
        self.stencil.set_backend (backend)
        self._run ( )

        if deps is None:
            deps = [ self.params ]

        stencil_deps = self.stencil.scope.data_dependency.edges ( )
        #
        # check the data-dependency detection for the whole stencil
        #
        while len (deps) > 0:
            first, second = deps.pop ( )
            found         = False
            for sd in stencil_deps:
                if first == sd[0].name and second == sd[1].name:
                    found = True
                    break
            if not found:
                self.assertTrue (False,
                                 "Dependency <%s,%s> not found in %s" %
                                 (first, second, stencil_deps))


    @attr(lang='cuda')
    def test_data_dependency_detection_cuda (self, deps=None, backend='cuda'):
        self.test_data_dependency_detection (deps=deps,
                                             backend=backend)


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS

        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_cpy',  None)
        self.add_expected_offset ('out_cpy', None)

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    @attr(lang='c++')
    def test_compare_python_and_cpp_results (self):
        import copy

        nruns                  = 5
        ndiff                  = 0
        stencil_native         = copy.deepcopy (self.stencil)
        stencil_native.set_backend ('c++')

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
        for i in range (nruns):
            self.stencil.run   (**params_py)
            stencil_native.run (**params_cxx)
            #
            # compare field contents
            #
            for k in params_py.keys ( ):
                diff  = np.isclose(params_py[k], params_cxx[k], atol=1e-11)
                ndiff += np.count_nonzero (np.logical_not (diff))
        #
        # Print statements for debugging purposes
        #
#        for i in range(self.domain[0]):
#            for j in range(self.domain[1]):
#                for k in range(self.domain[2]):
#                    print ("PY  (%d,%d,%d) \t%.5f \t%.5f" % (i,j,k,params_py['in_X'][i,j,k],
#                           params_py['out_X'][i,j,k]) )
#                    print ("CPP (%d,%d,%d) \t%.5f \t%.5f" % (i,j,k,params_cxx['in_X'][i,j,k],
#                           params_cxx['out_X'][i,j,k]) )
        print ("%s ndiff: %d" % ('c++', ndiff))
        print ("%d runs. Avg ndiff per run: %g." % (nruns, ndiff/nruns))
        self.assertEqual (ndiff, 0)


    @attr(lang='cuda')
    def test_compare_python_and_cuda_results (self):
        import copy

        nruns                  = 5
        ndiff                  = 0
        stencil_native         = copy.deepcopy (self.stencil)
        stencil_native.set_backend ('cuda')

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
        for i in range (nruns):
            self.stencil.run   (**params_py)
            stencil_native.run (**params_cxx)
            #
            # compare field contents
            #
            for k in params_py.keys ( ):
                diff  = np.isclose(params_py[k], params_cxx[k], atol=1e-11)
                ndiff += np.count_nonzero (np.logical_not (diff))
        #
        # Print statements for debugging purposes
        #
#        for i in range(self.domain[0]):
#            for j in range(self.domain[1]):
#                for k in range(self.domain[2]):
#                    print ("PY  (%d,%d,%d) \t%.5f \t%.5f" % (i,j,k,params_py['in_X'][i,j,k],
#                           params_py['out_X'][i,j,k]) )
#                    print ("CUDA (%d,%d,%d) \t%.5f \t%.5f" % (i,j,k,params_cxx['in_X'][i,j,k],
#                           params_cxx['out_X'][i,j,k]) )
        print ("%s ndiff: %d" % ('cuda', ndiff))
        print ("%d runs. Avg ndiff per run: %g." % (nruns, ndiff/nruns))
        self.assertEqual (ndiff, 0)


    def test_ghost_cell_pattern (self, expected_patterns=None, backend='c++'):
        self.stencil.set_backend (backend)
        self._run ( )

        if expected_patterns is None:
            expected_patterns = [ [0,0,0,0] ]

        self.assertEqual (len (self.stencil.stages), len (expected_patterns),
                          "Found %d stages, but %d ghost-cell patterns were given" %
                          (len (self.stencil.stages), len (expected_patterns)))
        for idx in range (len (self.stencil.stages)):
            self.assertEqual (self.stencil.stages[idx].ghost_cell,
                              expected_patterns[idx])


    def test_minimum_halo_detection (self, min_halo=[0,0,0,0]):
        from random    import randint
        from gridtools import BACKENDS

        #
        # minimum halo correctly calculated
        #
        for back in BACKENDS:
            self.stencil.set_backend (back)
            self._run ( )
            self.assertEqual (self.stencil.scope.minimum_halo,
                              min_halo)
            #
            # exception raised in case the provided halo is negative or smaller
            # than the minimum required
            #
            bad_halo = list (min_halo)
            for idx in range (len (min_halo)):
                bad_halo[idx] -= randint (1, 2)
            with self.assertRaises (ValueError):
                self.stencil.set_halo (bad_halo)
                self._run ( )
            #
            # execute normally in case the provided halo is bigger
            #
            big_halo = list (min_halo)
            for idx in range (len (min_halo)):
                big_halo[idx] += randint (0, 5)
            self.stencil.set_halo (big_halo)
            self._run ( )


    def test_symbol_discovery (self, backend='c++'):
        self.stencil.set_backend (backend)
        self._run ( )
        #
        # check fields were correctly discovered
        #
        scope = self.stencil.scope
        for p in self.params:
            self.assertTrue (scope.is_parameter (p))
        for t in self.temps:
            self.assertTrue (scope.is_temporary (t))


    @attr(lang='cuda')
    def test_symbol_discovery_cuda (self):
        self.test_symbol_discovery (backend='cuda')


    def test_user_stencil_extends_multistagestencil (self):
        from gridtools.stencil import Stencil

        with self.assertRaises (TypeError):
            class DoesNotExtendAndShouldFail (object):
                pass
            Stencil.compiler.register (DoesNotExtendAndShouldFail ( ))


    def test_run_stencil_only_accepts_keyword_arguments (self):
        with self.assertRaises (KeyError):
            self.stencil.run ([ getattr (self, p) for p in self.params ])


    @attr(lang='python')
    def test_python_results (self, out_param=None, result_file=None):
        import os

        cur_dir = os.path.dirname (os.path.abspath (__file__))

        self.stencil.set_backend ('python')
        self._run ( )
        #
        # take halo into account when comparing the results
        #
        beg_i = self.stencil.get_halo ( ) [0]
        end_i = self.domain[0] - self.stencil.get_halo ( ) [1]
        beg_j = self.stencil.get_halo ( ) [2]
        end_j = self.domain[1] - self.stencil.get_halo ( ) [3]

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

        self.stencil.set_backend (backend)
        self._run ( )

        nstep  = 100
        tstart = time.time ( )
        for i in range (nstep):
            self._run ( )
        print ('FPS:', nstep / (time.time()-tstart))


    @attr(lang='cuda')
    def test_execution_performance_cuda (self):
        self.test_execution_performance_cpp (backend='cuda')


    def test_k_directions (self, backend='c++'):
        self.stencil.set_backend (backend)
        for dir in ('forward', 'backward'):
            self.stencil.set_k_direction (dir)
            self._run ( )


    @attr(lang='cuda')
    def test_k_directions_cuda (self):
        self.test_k_directions (backend='cuda')


    def test_user_kernel_call (self):
        with self.assertRaises (RuntimeError):
            self.stencil.kernel (self.out_cpy, self.in_cpy)


    def test_decorator_returned_type (self):
        #
        # Test that the decorator returned a function method and not a
        # UserStencil object
        #
        import inspect
        self.assertTrue (inspect.ismethod (self.stencil.kernel))


    def test_get_halo_from_Stencil (self):
        from random import randint
        #
        # Set a new global halo and reset the stencil-specific value
        #
        new_halo = (randint(0,10), randint(0,10), randint(0,10), randint(0,10))
        Stencil.set_halo (new_halo)
        self.stencil.set_halo (None)
        #
        # Check that the global halo is correctly returned
        #
        self.assertEqual (Stencil.get_halo ( ), new_halo)
        self.assertEqual (self.stencil.get_halo ( ), new_halo)


    def test_get_halo_from_object (self):
        from random import randint
        #
        # Set a new global halo and a new stencil-specific value
        #
        global_halo = (randint(0,10), randint(0,10), randint(0,10), randint(0,10))
        Stencil.set_halo (global_halo)
        new_halo = (randint(0,10), randint(0,10), randint(0,10), randint(0,10))
        self.stencil.set_halo (new_halo)
        #
        # Check that the stencil halo is returned
        #
        self.assertEqual (Stencil.get_halo ( ), global_halo)
        self.assertEqual (self.stencil.get_halo ( ), new_halo)


    def test_get_k_direction_from_Stencil (self):
        from gridtools import K_DIRECTIONS
        from random import choice
        #
        # Set a new global k direction and reset the stencil-specific value
        #
        direction = choice(K_DIRECTIONS)
        Stencil.set_k_direction (direction)
        self.stencil.set_k_direction (None)
        #
        # Check that the global k direction is correctly returned
        #
        self.assertEqual (Stencil.get_k_direction ( ), direction)
        self.assertEqual (self.stencil.get_k_direction ( ), direction)


    def test_get_k_direction_from_object (self):
        from gridtools import K_DIRECTIONS
        from random import choice
        #
        # Set a new global k direction and a new stencil-specific value
        #
        global_direction = choice(K_DIRECTIONS)
        Stencil.set_k_direction(global_direction)

        direction = choice(K_DIRECTIONS)
        self.stencil.set_k_direction (direction)
        #
        # Check that the stencil k direction is correctly returned
        #
        self.assertEqual (Stencil.get_k_direction ( ),  global_direction)
        self.assertEqual (self.stencil.get_k_direction ( ), direction)


    def test_interior_points_generator (self):
        from gridtools import K_DIRECTIONS
        from random import randint, choice
        from itertools import product

        data_field = np.zeros(self.domain)

        halo = (randint(0,10), randint(0,10), randint(0,10), randint(0,10))
        direction = choice (K_DIRECTIONS)
        #
        # Generate slice indexes for all 3 array dimensions
        # We choose arbitrarily to limit the random numbers at 1/4th of the
        # domain size
        # Do not slice vertical direction if k<4 (happens in some tests,
        # ie Shallow Water)
        #
        slices = (randint(0, self.domain[0]//4), randint(self.domain[0]//4, self.domain[0]),
                  randint(0, self.domain[1]//4), randint(self.domain[1]//4, self.domain[1]),
                  randint(0, self.domain[2]//4), randint(self.domain[2]//4, self.domain[2]))
        #
        #
        # Generate expected coordinates
        #
        i_range = range (halo[0]+slices[0], slices[1]-halo[1])
        j_range = range (halo[2]+slices[2], slices[3]-halo[3])
        if direction == 'forward':
            k_range = range (slices[4], slices[5])
        else:
            k_range = range (slices[5]-1, slices[4]-1, -1)

        ijk_vals = list (product (i_range, j_range, k_range))
        #
        # Slice data field
        #
        sliced_field = data_field[slices[0]:slices[1],
                                  slices[2]:slices[3],
                                  slices[4]:slices[5]]
        #
        # Retrieve coordinates from function
        #
        interior_pts = list (Stencil._interior_points_generator (sliced_field,
                                                                 ghost_cell=[0,0,0,0],
                                                                 halo=halo,
                                                                 k_direction=direction))

        self.assertTrue (all ([x==y for (x,y) in zip(ijk_vals, interior_pts)]))


    def test_get_interior_points_static (self):
        from gridtools import K_DIRECTIONS
        from random import randint, choice
        from itertools import product

        data_field = np.zeros(self.domain)

        halo = (randint(0,10), randint(0,10), randint(0,10), randint(0,10))
        direction = choice (K_DIRECTIONS)
        Stencil.set_halo (halo)
        Stencil.set_k_direction (direction)
        #
        # Generate expected coordinates
        #
        i_range = range (halo[0], self.domain[0]-halo[1])
        j_range = range (halo[2], self.domain[1]-halo[3])
        if direction == 'forward':
            k_range = range (self.domain[2])
        else:
            k_range = range (self.domain[2]-1, -1, -1)

        ijk_vals = list (product (i_range, j_range, k_range))
        #
        # Retrieve coordinates from function
        #
        interior_pts = list (Stencil.get_interior_points (data_field))

        self.assertTrue (all ([x==y for (x,y) in zip(ijk_vals, interior_pts)]))


    def test_get_interior_points_object (self):
        from gridtools import K_DIRECTIONS
        from random import randint, choice
        from itertools import product

        data_field = np.zeros(self.domain)

        Stencil.set_halo ( (randint(0,10), randint(0,10), randint(0,10), randint(0,10)) )
        Stencil.set_k_direction ( choice(K_DIRECTIONS) )

        halo = ( randint(0,10), randint(0,10), randint(0,10), randint(0,10) )
        direction = choice (K_DIRECTIONS)
        self.stencil.set_halo (halo)
        self.stencil.set_k_direction (direction)
        #
        # Generate expected coordinates
        #
        i_range = range (halo[0], self.domain[0]-halo[1])
        j_range = range (halo[2], self.domain[1]-halo[3])
        if direction == 'forward':
            k_range = range (self.domain[2])
        else:
            k_range = range(self.domain[2]-1, -1, -1)

        ijk_vals = list (product (i_range, j_range, k_range))
        #
        # Retrieve coordinates from function
        #
        interior_pts = list (self.stencil.get_interior_points (data_field))

        self.assertTrue (all ([x==y for (x,y) in zip(ijk_vals, interior_pts)]) )


    def test_enforce_optimal_array (self):
        #
        # Convert arrays to C order
        #
        self.in_cpy = np.ascontiguousarray (self.in_cpy)
        self.out_cpy = np.ascontiguousarray (self.out_cpy)
        #
        # Apply memory layout conversion
        #
        self.in_cpy = self.stencil.compiler.utils.enforce_optimal_array (self.in_cpy,
                                                                         name='in_cpy',
                                                                         backend='cuda')
        self.out_cpy = self.stencil.compiler.utils.enforce_optimal_array (self.out_cpy,
                                                                         name='out_cpy',
                                                                         backend='cuda')
        #
        # Check that arrays have been converted to Fortran order
        #
        self.assertTrue (self.in_cpy.flags['F_CONTIGUOUS'])
        self.assertTrue (self.out_cpy.flags['F_CONTIGUOUS'])



class AnyKernelName (MultiStageStencil):
    """
    Imitates the CopyStencil using a different kernel name
    """
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
    def entry_point (self, out_cpy, in_cpy):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_cpy):
              out_cpy[p] = in_cpy[p]



class AnyKernelNameTest (CopyTest):
    """
    Tests that entry point function can have any name
    """
    def setUp (self):
        super ( ).setUp ( )
        self.stencil = AnyKernelName ( )
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_user_kernel_call (self):
        with self.assertRaises (RuntimeError):
            self.stencil.entry_point(self.out_cpy, self.in_cpy)


    def test_decorator_returned_type (self):
        import inspect
        self.assertTrue (inspect.ismethod (self.stencil.entry_point))



class FloatPrecisionTest (CopyTest):
    """
    Tests for the exceptions raised by the floating point precision validation.
    """
    def test_float_input_type_validation_wrong_data_size (self):
        with self.assertRaises (TypeError):
            self.stencil.set_backend ('c++')
            self.in_cpy = self.in_cpy.astype(np.float16)
            self._run ( )


    def test_float_input_type_validation_wrong_data_type (self):
        with self.assertRaises (TypeError):
            self.stencil.set_backend ('c++')
            self.in_cpy = self.in_cpy.astype(np.int64)
            self._run ( )


    def test_float_input_type_validation_potentially_correct_type_but_not_the_correct_size (self):
        with self.assertRaises (TypeError):
            self.stencil.set_backend ('c++')
            self.in_cpy = self.in_cpy.astype(np.float32)
            self._run ( )



class Power (MultiStageStencil):
    """
    Imitates the CopyStencil using the power operator.-
    """
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
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
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")



class Laplace (MultiStageStencil):
    """
    A Laplacian operator, as the one used in COSMO.-
    """
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
    def kernel (self, out_data, in_data):
        """
        Stencil's entry point.-
        """
        for p in self.get_interior_points (out_data):
            out_data[p] = -4.0 * in_data[p] + (
                          in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                          in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )



class LaplaceTest (CopyTest):
    """
    Testing the Laplace operator.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.domain = (64, 64, 32)
        self.params = ('out_data', 'in_data')
        self.temps  = ( )

        self.out_data = np.zeros (self.domain, order='F')
        self.in_data  = np.zeros (self.domain, order='F')
        for i in range (self.domain[0]):
            for j in range (self.domain[1]):
                for k in range (self.domain[2]):
                    self.in_data[i,j,k] = i**3 + j

        self.stencil = Laplace ( )
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_data_dependency_detection (self, expected_deps=None, backend='c++'):
        if expected_deps is None:
            expected_deps = [('out_data', 'in_data')]
        super ( ).test_data_dependency_detection (deps=expected_deps,
                                                  backend=backend)


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS
        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_data',  [-1,1,-1,1])
        self.add_expected_offset ('out_data', None)

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_minimum_halo_detection (self, min_halo=None):
        if min_halo is None:
            min_halo = [1, 1, 1, 1]
        super ( ).test_minimum_halo_detection (min_halo)


    @attr(lang='python')
    def test_python_results (self):
        self.out_data = np.random.rand (*self.domain)
        super ( ).test_python_results (out_param='out_data',
                                       result_file='laplace_result.npy')



class LocalLaplace (MultiStageStencil):
    """
    A Laplacian operator, as the one used in COSMO.-
    """
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
    def kernel (self, out_data, in_data):
        """
        Stencil's entry point.-
        """
        for p in self.get_interior_points (out_data):
            out_value   = -4.0 * in_data[p] + (
                          in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                          in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )
            out_data[p] = out_value



class LocalLaplaceTest (LaplaceTest):
    """
    Testing the LocalLaplace stencil.-
    """
    def setUp (self):
        super ( ).setUp ( )

        self.stencil = LocalLaplace ( )
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS
        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_data',  [-1,1,-1,1])
        self.add_expected_offset ('out_value', None)
        self.add_expected_offset ('out_data', None)

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)



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
        for p in self.get_interior_points (out_fli,
                                           ghost_cell=[-1,0,-1,0]):
            out_fli[p] = in_lap[p + (1,0,0)] - in_lap[p]


    def stage_flux_j (self, out_flj, in_lap):
        for p in self.get_interior_points (out_flj,
                                           ghost_cell=[-1,0,-1,0]):
            out_flj[p] = in_lap[p + (0,1,0)] - in_lap[p]


    @Stencil.kernel
    def kernel (self, out_data, in_data, in_wgt):
        #
        # Laplace
        #
        for p in self.get_interior_points (self.lap,
                                           ghost_cell=[-1,1,-1,1]):
            self.lap[p] = -4.0 * in_data[p] +  (
                          in_data[p + (-1,0,0)] + in_data[p + (1,0,0)] +
                          in_data[p + (0,-1,0)] + in_data[p + (0,1,0)] )
        #
        # the fluxes are independent, because they depend on 'self.lap'
        #
        self.stage_flux_i (out_fli = self.fli,
                           in_lap  = self.lap)
        self.stage_flux_j (out_flj = self.flj,
                           in_lap  = self.lap)

        for p in self.get_interior_points (out_data):
            #
            # Last stage
            #
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

        self.out_data = np.zeros (self.domain, order='F')
        self.in_wgt   = np.ones  (self.domain, order='F')
        self.in_data  = np.zeros (self.domain, order='F')

        for i in range (self.domain[0]):
            for j in range (self.domain[1]):
                for k in range (self.domain[2]):
                    self.in_data[i,j,k] = i**5 + j

        self.stencil = HorizontalDiffusion (self.domain)
        self.stencil.set_halo ( (2, 2, 2, 2) )
        self.stencil.set_k_direction ("forward")


    def test_data_dependency_detection (self, expected_deps=None, backend='c++'):
        if expected_deps is None:
            expected_deps = [('out_data', 'in_wgt'),
                             ('out_data', 'self.flj'),
                             ('out_data', 'self.fli'),
                             ('self.fli', 'self.lap'),
                             ('self.flj', 'self.lap'),
                             ('self.lap', 'in_data')]
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
        self.add_expected_offset ('in_data',  [-1,1,-1,1])
        self.add_expected_offset ('in_wgt',   None)
        self.add_expected_offset ('out_data', None)
        self.add_expected_offset ('self.fli', None)
        self.add_expected_offset ('self.fli', [-1,0,0,0])
        self.add_expected_offset ('self.flj', None)
        self.add_expected_offset ('self.flj', [0,0,-1,0])
        self.add_expected_offset ('self.lap', None)
        self.add_expected_offset ('self.lap', [0,1,0,0])
        self.add_expected_offset ('self.lap', [0,0,0,1])

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_ghost_cell_pattern (self, expected_patterns=None, backend='c++'):
        if expected_patterns is None:
            expected_patterns = [ [-1,1,-1,1],
                                  [-1,0,-1,0],
                                  [-1,0,-1,0],
                                    [0,0,0,0] ]
        super ( ).test_ghost_cell_pattern (expected_patterns,
                                           backend=backend)

    @attr(lang='cuda')
    def test_ghost_cell_pattern_cuda (self):
        self.test_ghost_cell_pattern (backend='cuda')


    def test_minimum_halo_detection (self, min_halo=None):
        if min_halo is None:
            min_halo = [2, 2, 2, 2]
        super ( ).test_minimum_halo_detection (min_halo)


    @attr(lang='python')
    def test_python_results (self):
        self.out_data = np.random.rand (*self.domain)
        super ( ).test_python_results (out_param='out_data',
                                       result_file='horizontaldiffusion_result.npy')



class VerticalRegions (MultiStageStencil):
    """
    A stencil using a Laplacian-like operator with different vertical regions
    Notable features:
    * Vertical regions overlap between all stages: stencil splitter ordering is
        not trivial
    * stage_laplace3 has a vertical region corresponding with the end of
        stage_laplace0 and the start of stage_laplace2: this will generate
        duplicate splitters that have to be eliminated at stencil level
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.domain = domain


    def stage_laplace0 (self, out_data, in_data):
        for p in self.get_interior_points (out_data[:,:,0:4]):
            out_data[p] = -4.0 * in_data[p] + (
                          in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                          in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


    def stage_laplace1 (self, out_data, in_data):
        for p in self.get_interior_points (out_data[:,:,3:8]):
            out_data[p] = -6.0 * in_data[p] + (
                          in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                          in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


    def stage_laplace2 (self, out_data, in_data):
        for p in self.get_interior_points (out_data[:,:,6:]):
            out_data[p] = -8.0 * in_data[p] + (
                          in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                          in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


    def stage_laplace3 (self, out_data, in_data):
        for p in self.get_interior_points (out_data[:,:,4:8]):
            out_data[p] = -10.0 * in_data[p] + (
                          in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                          in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


    @Stencil.kernel
    def kernel (self, out_data0, out_data1, out_data2, out_data3, in_data):
        self.stage_laplace0 (out_data = out_data0,
                            in_data = in_data)
        self.stage_laplace1 (out_data = out_data1,
                            in_data = in_data)
        self.stage_laplace2 (out_data = out_data2,
                            in_data = in_data)
        self.stage_laplace3 (out_data = out_data3,
                            in_data = in_data)



class VerticalRegionsTest (LaplaceTest):
    """
    Test fixture for the VerticalRegions stencil defined above

    TODO: Add tests for:
            - overlapping vertical regions within the same stage (requires
                support for multiple vertical regions within a stage),
            - negative slicing at the beggining or end,
            - positive slicing at the beggining with field overflow,
            - positive slicing at the end with field overflow,
            - using a variable/attribute instead of a constant when slicing.

    """
    def setUp (self):
        super ( ).setUp ( )

        self.params = ('out_data0',
                       'out_data1',
                       'out_data2',
                       'out_data3',
                       'in_data')

        self.out_data0 = np.zeros (self.domain, order='F')
        self.out_data1 = np.zeros (self.domain, order='F')
        self.out_data2 = np.zeros (self.domain, order='F')
        self.out_data3 = np.zeros (self.domain, order='F')

        self.stencil = VerticalRegions (self.domain)
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_data_dependency_detection (self, expected_deps=None, backend='c++'):
        if expected_deps is None:
            expected_deps = [('out_data0', 'in_data'),
                             ('out_data1', 'in_data'),
                             ('out_data2', 'in_data'),
                             ('out_data3', 'in_data')]
        super ( ).test_data_dependency_detection (expected_deps=expected_deps,
                                                  backend=backend)


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS
        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_data', None)
        self.add_expected_offset ('in_data', None)
        self.add_expected_offset ('in_data', None)
        self.add_expected_offset ('in_data', None)
        self.add_expected_offset ('in_data',  [-1,1,-1,1])
        self.add_expected_offset ('in_data',  [-1,1,-1,1])
        self.add_expected_offset ('in_data',  [-1,1,-1,1])
        self.add_expected_offset ('in_data',  [-1,1,-1,1])
        self.add_expected_offset ('out_data0', None)
        self.add_expected_offset ('out_data1', None)
        self.add_expected_offset ('out_data2', None)
        self.add_expected_offset ('out_data3', None)

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_ghost_cell_pattern (self, expected_patterns=None, backend='c++'):
        if expected_patterns is None:
            expected_patterns = [ [0,0,0,0],
                                  [0,0,0,0],
                                  [0,0,0,0],
                                  [0,0,0,0] ]
        super ( ).test_ghost_cell_pattern (expected_patterns,
                                           backend=backend)


    def test_minimum_halo_detection (self, min_halo=None):
        if min_halo is None:
            min_halo = [1, 1, 1, 1]
        super ( ).test_minimum_halo_detection (min_halo)


    @unittest.skip("Not yet implemented")
    @attr(lang='python')
    def test_python_results (self):
        pass


    def test_splitters (self):
        self._run ( )
        self.assertEqual (self.stencil.splitters,
                          {0: 0, 3: 1, 4: 2, 6: 3, 8: 4, 32: 5})


    def test_vertical_regions (self):
        self._run ( )
        expected_vr = {'stage_laplace0': [0,4],
                       'stage_laplace1': [3,8],
                       'stage_laplace2': [6,32],
                       'stage_laplace3': [4,8]}
        #
        # Iterate over stages, popping expected values from the dictionary
        # A dict is required because stages is the result of a topological sort,
        # so we don't know the order the stages will be presented to us.
        #
        for stg in self.stencil.stages:
            #
            # Check only the stage-specific part of the name string, as we don't
            # know the name of the stencil and the number that will be prepended
            # or appended to the name string at runtime
            #
            vr_key = None
            for k in expected_vr.keys():
                if k in stg.name:
                    vr_key = k
            #
            # Check that VR edges correspond to expected ones
            #
            vr_edges = expected_vr.pop(vr_key)
            self.assertEqual (stg.vertical_regions[0].start_splitter, vr_edges[0])
            self.assertEqual (stg.vertical_regions[0].end_splitter, vr_edges[1])
        #
        # Check all expected vertical regions have been tested (dict is empty)
        #
        self.assertFalse (expected_vr)



class ChildStencilCallsParentConstructorAndNothingElse (MultiStageStencil):
    """
    Child constructor correctly calls parent constructor and has no other work, comments or docstrings.
    Works correctly--no exceptions.
    """
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
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


    @Stencil.kernel
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


    @Stencil.kernel
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


    @Stencil.kernel
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


    @Stencil.kernel
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
        pass



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
    def setUp (self):
        logging.basicConfig (level=logging.INFO)

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


    def _test_child_constructor_call_success (self, stencil):
        from gridtools.stencil import Stencil

        kwargs = dict ( )
        for p in self.params:
            kwargs[p] = getattr (self, p)
        Stencil.compiler.analyze (stencil, **kwargs)


    def test_child_constructor_calls_parent_constructor_and_nothing_else (self):
        self._test_child_constructor_call_success (ChildStencilCallsParentConstructorAndNothingElse ( ))


    def test_child_constructor_calls_parent_constructor_first (self):
        self._test_child_constructor_call_success (ChildStencilCallsParentConstructorFirst ( ))


    def test_child_constructor_calls_parent_constructor_after_comment (self):
        self._test_child_constructor_call_success (ChildStencilCallsParentConstructorAfterComment ( ))


    def test_child_constructor_calls_parent_constructor_after_mult_comments (self):
        self._test_child_constructor_call_success (ChildStencilCallsParentConstructorAfterMultComments ( ))


    def test_child_constructor_calls_parent_constructor_after_doc_string (self):
        self._test_child_constructor_call_success (ChildStencilCallsParentConstructorAfterDocString ( ))


    def _test_child_constructor_call_fails (self, stencil):
        with self.assertRaises (Exception):
            self._test_child_constructor_call_success (stencil)


    def test_child_constructor_calls_parent_constructor_after_mult_doc_strings (self):
        self._test_child_constructor_call_fails (ChildStencilCallsParentConstructorAfterMultDocStrings ( ))


    def test_child_constructor_no_call_parent_constructor (self):
        self._test_child_constructor_call_fails (ChildStencilNoCallParentConstructor ( ))


    def test_child_constructor_calls_parent_constructor_after_assignment (self):
        self._test_child_constructor_call_fails (ChildStencilCallsParentConstructorAfterAssignment ( ))


    def test_child_constructor_calls_parent_constructor_after_string_assignment (self):
        self._test_child_constructor_call_fails (ChildStencilCallsParentConstructorAfterStringAssignment ( ))


    def test_child_constructor_calls_parent_constructor_after_computation (self):
        self._test_child_constructor_call_fails (ChildStencilParentConstructorAfterComputation ( ))
