import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools import MultiStageStencil, StencilInspector




class Copy (MultiStageStencil):
    """
    Definition of a simple copy stencil, as in 'examples/copy_stencil.h'.-
    """
    def __init__ (self):
        super (Copy, self).__init__ ( )

    def kernel (self, out_data, in_data):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_data):
              out_data[p] = in_data[p]



class CopyTest (unittest.TestCase):
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

        self.domain = (16, 16, 8)
        self.params = ('out_data', 'in_data')
        self.temps  = ( )

        self.out_data = np.zeros (self.domain)
        self.in_data  = np.random.rand (*self.domain)

        self.stencil = Copy ( )
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_automatic_range_detection (self, ranges=None):
        """
        Parameter 'ranges' is a dictionary where the key is the name of a
        data field and its value the expected range.-
        """
        self.stencil.backend = 'c++'
        self._run ( )

        #
        # check the range detection within each functor
        #
        expected = None
        for f in self.stencil.inspector.functors:
            sc = f.scope
            for p in sc.get_all ( ):
                try:
                    expected = ranges[p.name]
                except (TypeError, KeyError):
                    expected = None
                self.assertEqual (sc[p.name].range, expected)


    def test_compare_python_and_native_executions (self):
        import copy

        stencil_native         = copy.deepcopy (self.stencil)
        stencil_native.backend = 'c++'

        #
        # data fields - Py and C++ sets
        #
        params_py  = dict ( )
        params_cxx = dict ( )
        for p in self.params:
            params_py[p]  = np.random.rand (*self.domain)
            params_cxx[p] = np.copy (params_py[p])

        #
        # apply both stencils 10 times
        #
        for i in range (10):
            self.stencil.run   (**params_py)
            stencil_native.run (**params_cxx)

            #
            # compare the field contents
            #
            for k in params_py.keys ( ):
                self.assertTrue (np.array_equal (params_py[k],
                                                 params_cxx[k]))


    def test_symbol_discovery (self):
        self.stencil.backend = 'c++'
        self._run ( )
        #
        # check input/output fields were correctly discovered
        #
        scope = self.stencil.inspector.stencil_scope
        for p in self.params:
            self.assertTrue (scope.is_parameter (p))
        for t in self.temps:
            self.assertTrue (scope.is_temporary (t))


    def test_user_stencil_extends_multistagestencil (self):
        with self.assertRaises (TypeError):
            class DoesNotExtendAndShouldFail (object):
                pass
            insp = StencilInspector (DoesNotExtendAndShouldFail)

        insp = self.stencil.inspector
        insp.analyze ( )
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
    def test_python_execution (self):
        self._run ( )

        beg_i = self.stencil.halo[0]
        end_i = self.domain[0] - self.stencil.halo[1]
        beg_j = self.stencil.halo[2]
        end_j = self.domain[1] - self.stencil.halo[3]

        self.assertTrue (np.array_equal (self.in_data[beg_i:end_i, beg_j:end_j],
                                         self.out_data[beg_i:end_i, beg_j:end_j]))


    @attr(lang='c++')
    def test_native_execution_performance (self):
        import time

        self.stencil.backend = 'c++'
        self._run ( )
        self.assertTrue ('_FuncPtr' in dir (self.stencil.lib_obj))

        avg_time = 0.0
        for i in range (30):
            start = time.time ( )
            self._run ( )
            avg_time += time.time ( ) - start
        print ("AVG execution time %.3f s" % avg_time)


    def test_k_directions (self):
        self.stencil.backend = 'c++'

        for dir in ('forward', 'backward'):
            self.stencil.set_k_direction (dir)
            self._run ( )


class Laplace (MultiStageStencil):
    """
    A Laplacian operator, as the one used in COSMO.-
    """
    def __init__ (self):
        super ( ).__init__ ( )


    def kernel (self, out_data, in_data):
        """
        Stencil's entry point.-
        """
        #
        # iterate over the interior points
        #
        for p in self.get_interior_points (out_data):
            out_data[p] = 4 * in_data[p] - (
                          in_data[p + (1,0,0)] + in_data[p + (0,1,0)] +
                          in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )



class LaplaceTest (CopyTest):
    """
    Testing the Laplace operator.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.INFO)

        self.domain = (64, 64, 32)
        self.params = ('out_data', 'in_data')
        self.temps  = ( )

        self.out_data  = np.zeros (self.domain)
        self.in_data   = np.arange (np.prod (self.domain)).reshape (self.domain)
        self.in_data  /= 7.0

        self.stencil = Laplace ( ) 
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_automatic_range_detection (self):
        expected_ranges = {'out_data': None,
                           'in_data':  [-1,1,-1,1]}
        super ( ).test_automatic_range_detection (ranges=expected_ranges)


    @attr(lang='python')
    def test_python_execution (self):
        import os

        self.stencil.backend = 'python'
        self._run ( )

        cur_dir  = os.path.dirname (os.path.abspath (__file__))
        expected = np.load ('%s/laplace_result.npy' % cur_dir)

        self.assertTrue (np.array_equal (self.out_data,
                                         expected))




class HorizontalDiffusion (MultiStageStencil):
    def __init__ (self, domain):
        super ( ).__init__ ( )

        #
        # temporary data fields to share data among the different stages
        #
        self.lap = np.zeros (domain)
        self.fli = np.zeros (domain)
        self.flj = np.zeros (domain)


    def kernel (self, out_data, in_data, in_wgt):
        #
        # Laplace
        #
        for p in self.get_interior_points (in_data):
            self.lap[p] = -4.0 * in_data[p] +  (
                          in_data[p + (-1,0,0)] + in_data[p + (1,0,0)] +
                          in_data[p + (0,-1,0)] + in_data[p + (0,1,0)] )
        #
        # Flux over 'i'
        #
        for p in self.get_interior_points (self.lap):
            self.fli[p] = self.lap[p + (1,0,0)] - self.lap[p]
        #
        # Flux over 'j'
        #
        for p in self.get_interior_points (self.lap):
            self.flj[p] = self.lap[p + (0,1,0)] - self.lap[p]
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
        logging.basicConfig (level=logging.DEBUG)

        self.domain = (8, 8, 1)
        self.params = ('out_data', 
                       'in_data',
                       'in_wgt')
        self.temps  = ('self.lap', 
                       'self.fli',
                       'self.flj')

        self.out_data = np.zeros (self.domain)
        self.in_data  = np.random.rand (*self.domain)
        self.in_wgt   = np.ones  (self.domain)

        self.stencil = HorizontalDiffusion (self.domain)
        self.stencil.set_halo ( (2, 2, 2, 2) )
        self.stencil.set_k_direction ("forward")


    def test_automatic_range_detection (self):
        expected_ranges = {'out_data': None,
                           'in_data' : [-1,1,-1,1],
                           'in_wgt'  : None,
                           'lap'     : [0,1,0,0],
                           'fli'     : [-1,0,0,0],
                           'flj'     : [0,0,-1,0]}
        super ( ).test_automatic_range_detection (ranges=expected_ranges)


    @attr(lang='python')
    def test_python_execution (self):
        import os

        self.stencil.backend = 'python'
        self._run ( )
        print ("In_data:", np.sum (self.in_data))
        print ("lap tmp:", np.sum (self.stencil.lap))
        print ("fli tmp:", np.sum (self.stencil.fli))
        print ("flj tmp:", np.sum (self.stencil.flj))

        #
        # FIXME create the test data for this test
        #
        #cur_dir  = os.path.dirname (os.path.abspath (__file__))
        #expected = np.load ('%s/laplace_result.npy' % cur_dir)
        #self.assertTrue (np.array_equal (self.out_data,
        #                                 expected))



class Moving (MultiStageStencil):
    """
    Definition a stencil that should move the data over the domain.-
    """
    def __init__ (self, domain):
        """
        A comment to make AST parsing more difficult.-
        """
        super (Moving, self).__init__ ( )
        self.domain = domain
        #
        # grid size with a halo of one
        #
        self.n = domain[0] - 2

        #
        # gravity-accelleration constant
        #
        self.g = 9.8

        #
        # timestep
        #
        self.dt = 0.02

        #
        # space step size (for u, v)
        #
        self.dx = 1.0
        self.dy = 1.0

        #
        # temporary data fields
        #
        self.Hx = np.zeros ((self.n+1, self.n+1, 1))
        self.Ux = np.zeros ((self.n+1, self.n+1, 1))
        self.Vx = np.zeros ((self.n+1, self.n+1, 1))

        self.Hy = np.zeros ((self.n+1, self.n+1, 1))
        self.Uy = np.zeros ((self.n+1, self.n+1, 1))
        self.Vy = np.zeros ((self.n+1, self.n+1, 1))


    def droplet (self, height, width):
        """
        A two-dimensional Gaussian of the falling drop into the water:

            height  height of the generated drop;
            width   width of the generated drop.-
        """
        x = np.array ([np.arange (-1, 1 + 2/(width-1), 2/(width-1))] * (width-1))
        y = np.copy (x)
        drop = height * np.exp (-5*(x*x + y*y))
        #
        # pad the resulting array with zeros
        #
        #zeros = np.zeros (shape[:-1])
        #zeros[:drop.shape[0], :drop.shape[1]] = drop
        #return zeros.reshape (zeros.shape[0],
        #                      zeros.shape[1],
        #                      1)
        return drop.reshape ((drop.shape[0],
                              drop.shape[1],
                              1))


    def create_random_drop (self, H):
        """
        Disturbs the water surface with a drop.-
        """
        drop = self.droplet (5, 11)
        w = drop.shape[0]

        rand0 = np.random.rand ( )
        rand1 = np.random.rand ( )
        rand2 = np.random.rand ( )

        for i in range (w):
            i_idx = int (i + np.ceil (rand0 * (self.n - w)))
            for j in range (w):
                j_idx = int (j + np.ceil (rand1 * (self.n - w)))
                H[i_idx, j_idx] += rand2 * drop[i, j]


    def reflect_borders (self, H, U, V):
        """
        Implements the reflective boundary conditions in NumPy.-
        """
        H[:,0] =  H[:,1]
        U[:,0] =  U[:,1]
        V[:,0] = -V[:,1]

        H[:,self.n+1] =  H[:,self.n]
        U[:,self.n+1] =  U[:,self.n]
        V[:,self.n+1] = -V[:,self.n]

        H[0,:] =  H[1,:]
        U[0,:] = -U[1,:]
        V[0,:] =  V[1,:]

        H[self.n+1,:] =  H[self.n,:]
        U[self.n+1,:] = -U[self.n,:]
        V[self.n+1,:] =  V[self.n,:]


    def kernel (self, out_H, out_U, out_V):
        """
        This stencil comprises multiple stages.-
        """
        #
        # first half step (stage X direction)
        #
        for p in self.get_interior_points (self.Hx):
            # height
            self.Hx[p]  = out_H[p + (-1,-1,0)]

            # X momentum
            self.Ux[p]  = out_U[p + (-1,-1,0)]

            # Y momentum
            self.Vx[p]  = out_V[p + (-1,-1,0)]

        #
        # frst half step (stage Y direction)
        #
        for p in self.get_interior_points (self.Hy):
            # height
            self.Hy[p]  = out_H[p + (1,1,0)]

            # X momentum
            self.Uy[p]  = out_U[p + (1,1,0)]

            # Y momentum
            self.Vy[p]  = out_V[p + (1,1,0)]

        #
        # second half step (stage)
        #
        for p in self.get_interior_points (self.Hx):
            # height
            out_H[p] = self.Hx[p]

            # X momentum
            out_U[p] = self.Ux[p]

            # Y momentum
            out_V[p] = self.Vx[p]



class MovingTest (CopyTest):
    """
    A test case for the Moving stencil defined above.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.INFO)

        self.domain = (64, 64, 1)
        self.params = ('out_H', 
                       'out_U',
                       'out_V')
        self.temps  = ('self.Hx', 
                       'self.Hy',
                       'self.Ux',
                       'self.Uy',
                       'self.Vx',
                       'self.Vy')

        self.out_H = np.ones  (self.domain)
        self.out_U = np.zeros (self.domain)
        self.out_V = np.zeros (self.domain)

        self.stencil = Moving (self.domain)
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_compare_python_and_native_executions (self):
        pass

    def test_automatic_range_detection (self):
        self.stencil.backend = 'c++'
        self._run ( )

        exp_rng = [-1, 0, -1, 0]
        scope   = self.stencil.inspector.functors[0].scope
        self.assertEqual (scope['self.Hx'].range, None)
        self.assertEqual (scope['self.Ux'].range, None)
        self.assertEqual (scope['self.Vx'].range, None)
        self.assertEqual (scope['out_H'].range, exp_rng)
        self.assertEqual (scope['out_U'].range, exp_rng)
        self.assertEqual (scope['out_V'].range, exp_rng)

        exp_rng = [0, 1, 0, 1]
        scope   = self.stencil.inspector.functors[1].scope
        self.assertEqual (scope['self.Hy'].range, None)
        self.assertEqual (scope['self.Uy'].range, None)
        self.assertEqual (scope['self.Vy'].range, None)
        self.assertEqual (scope['out_H'].range, exp_rng)
        self.assertEqual (scope['out_U'].range, exp_rng)
        self.assertEqual (scope['out_V'].range, exp_rng)
        
        exp_rng = None
        scope   = self.stencil.inspector.functors[2].scope
        self.assertEqual (scope['self.Hx'].range, exp_rng)
        self.assertEqual (scope['self.Ux'].range, exp_rng)
        self.assertEqual (scope['self.Vx'].range, exp_rng)
        self.assertEqual (scope['out_H'].range, exp_rng)
        self.assertEqual (scope['out_U'].range, exp_rng)
        self.assertEqual (scope['out_V'].range, exp_rng)


    @attr(lang='python')
    def test_python_execution (self):
        import os

        self.stencil.backend = 'python'
        self._run ( )

        #cur_dir  = os.path.dirname (os.path.abspath (__file__))
        #expected = np.load ('%s/laplace_result.npy' % cur_dir)

        #self.assertTrue (np.array_equal (self.out_data,
        #                                 expected))

        
    def test_interactive_plot (self):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation, cm
            from mpl_toolkits.mplot3d import axes3d

            #
            # enable native execution for the stencil
            #
            self.stencil.backend = 'c++'

            #
            # disturb the water surface
            #
            self.stencil.create_random_drop (self.out_H)

            #
            # initialize 3D plot
            #
            fig = plt.figure ( )
            ax = axes3d.Axes3D (fig)

            rng  = np.arange (self.domain[0])
            X, Y = np.meshgrid (rng, rng)
            surf = ax.plot_wireframe (X, Y,
                                    np.squeeze (self.out_H, axis=(2,)),
                                    rstride=1,
                                    cstride=1,
                                    cmap=cm.jet,
                                    linewidth=1,
                                    antialiased=False)
            #
            # animation update function
            #
            def draw_frame (framenumber, swobj):
                #
                # a random drop
                #
                if framenumber == 0:
                    self.stencil.create_random_drop (self.out_H)

                #
                # reflective boundary conditions
                #
                swobj.reflect_borders (self.out_H,
                                       self.out_U,
                                       self.out_V)
                #
                # run the stencil
                #
                swobj.run (out_H=self.out_H,
                           out_U=self.out_U,
                           out_V=self.out_V)

                ax.cla ( )
                surf = ax.plot_wireframe (X, Y,
                                    np.squeeze (self.out_H, axis=(2,)),
                                    rstride=1,
                                    cstride=1,
                                    cmap=cm.jet,
                                    linewidth=1,
                                    antialiased=False)
                return surf,

            anim = animation.FuncAnimation (fig,
                                            draw_frame,
                                            fargs=(self.stencil,),
                                            frames=range (10),
                                            interval=20,
                                            blit=False)
        except ImportError:
            #
            # don't run this test if matplotlib is not available
            #
            pass


    def test_k_directions (self):
        self.stencil.backend = 'c++'

        for dir in ('forward'):
            self.stencil.set_k_direction (dir)
            self._run ( )

