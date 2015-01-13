import unittest
import logging

import numpy as np

from gridtools import MultiStageStencil, StencilInspector




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
        for p in self.get_interior_points (self.Hx,
                                           k_direction="forward"):
            # height
            self.Hx[p]  = out_H[p + (-1,-1,0)]

            # X momentum    
            self.Ux[p]  = out_U[p + (-1,-1,0)]

            # Y momentum
            self.Vx[p]  = out_V[p + (-1,-1,0)]

        #
        # first half step (stage Y direction)
        #
        for p in self.get_interior_points (self.Hy,
                                           k_direction="forward"):
            # height
            self.Hy[p]  = out_H[p + (1,1,0)]

            # X momentum    
            self.Uy[p]  = out_U[p + (1,1,0)]

            # Y momentum
            self.Vy[p]  = out_V[p + (1,1,0)]

        #
        # second half step (stage)
        #
        for p in self.get_interior_points (self.Hx,
                                           k_direction="forward"):
            # height
            out_H[p] = self.Hx[p]

            # X momentum
            out_U[p] = self.Ux[p]

            # Y momentum
            out_V[p] = self.Vx[p]


class MovingTest (unittest.TestCase):
    """
    A test case for the Moving stencil defined above.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.INFO)

        self.domain = (64, 64, 1)

        self.H = np.ones  (self.domain)
        self.U = np.zeros (self.domain)
        self.V = np.zeros (self.domain)

        self.stencil = Moving (self.domain)


    def test_interactive_plot (self):
        import matplotlib.pyplot as plt
        from matplotlib import animation, cm
        from mpl_toolkits.mplot3d import axes3d

        #
        # enable native execution for the stencil
        #
        #self.stencil.backend = 'c++'

        #
        # disturb the water surface
        #
        self.stencil.create_random_drop (self.H)

        #
        # initialize 3D plot
        #
        fig = plt.figure ( )
        ax = axes3d.Axes3D (fig)

        rng  = np.arange (self.domain[0])
        X, Y = np.meshgrid (rng, rng)
        surf = ax.plot_wireframe (X, Y, 
                                np.squeeze (self.H, axis=(2,)),
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
                self.stencil.create_random_drop (self.H)
    
            #
            # reflective boundary conditions
            #
            swobj.reflect_borders (self.H,
                                   self.U,
                                   self.V)
            #
            # run the stencil
            #
            swobj.run (out_H=self.H,
                       out_U=self.U,
                       out_V=self.V)

            ax.cla ( )
            surf = ax.plot_wireframe (X, Y, 
                                np.squeeze (self.H, axis=(2,)),
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
        for p in self.get_interior_points (out_data,
                                           k_direction="forward"):
              out_data[p] = in_data[p]



class CopyTests (unittest.TestCase):
    """
    A test case for the copy stencil defined above.-
    """
    def test_extends (self):
        """
        A user-defined stencil should inherit from the MultiStageStencil class.-
        """
        with self.assertRaises (TypeError):
            class DoesNotExtendAndShouldFail (object):
                pass
            insp = StencilInspector (DoesNotExtendAndShouldFail)

        insp = StencilInspector (Copy)
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


    def test_only_keyword_arguments (self):
        """
        When calling 'run' on a stencil, only keyword arguments should be used.-
        """
        domain = (128, 128, 60)
        output_field = np.zeros (domain)
        input_field = np.random.rand (*domain)
        copy = Copy ( )
        with self.assertRaises (KeyError):
            copy.run (output_field,
                      input_field)


    def test_python_execution (self):
        """
        Checks that the stencil results are correct if executing in Python mode.-
        """
        domain = (128, 128, 60)
        output_field = np.zeros (domain)
        input_field = np.random.rand (*domain)
        copy = Copy ( )
        copy.run (out_data=output_field,
                  in_data=input_field)
        self.assertTrue (np.array_equal (input_field, 
                                         output_field))


    def test_native_execution (self):
        """
        Checks stencil compilation and execution from a dynamic library.
        Note that the Python code is practically identical, except for the
        call to the 'backend' attribute.
        It also checks that the stencil results are correct after execution.-
        """
        domain = (512, 512, 60)
        output_field = np.zeros (domain)
        input_field  = np.random.rand (*domain)
        copy = Copy ( )
        copy.backend = 'c++'
        copy.run (out_data=output_field,
                  in_data=input_field)
        self.assertNotEqual (copy.inspector.lib_obj, None)
        self.assertTrue     ('_FuncPtr' in dir (copy.inspector.lib_obj))
        #
        # compare the arrays taking a 2D-halo of 1 into account
        #
        self.assertEqual (np.sum (input_field[1:511,1:511]),
                          np.sum (output_field[1:511,1:511]))



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
        for p in self.get_interior_points (out_data,
                                           halo=(1,-1,1,-1),
                                           k_direction="forward"):
            out_data[p] = 4 * in_data[p] - (
                          in_data[p + (1,0,0)] + in_data[p + (0,1,0)] +
                          in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


class LaplaceTests (unittest.TestCase):
    """
    Testing the Laplace operator.-
    """
    def test_compare_python_and_native_executions (self):
        """
        Checks that the stencil results match for Python and C++ after
        applying the stencil several times.-
        """
        domain = (64, 64, 64)

        lap_py = Laplace ( )
        lap_cxx = Laplace ( )
        lap_cxx.backend = 'c++'

        #
        # domain and fields
        #
        o_field = np.zeros (domain)
        i_field = np.random.rand (*domain)

        #
        # apply the Laplace operator 10 times
        #
        for i in range (10):
            #
            # original content of the data fields
            #
            orig_o = np.array (o_field)
            orig_i = np.array (i_field)

            #
            # apply the Python version of the stencil
            #
            lap_py.run (out_data=o_field,
                        in_data=i_field)
            #
            # apply the native version of the stencil
            #
            lap_cxx.run (out_data=orig_o,
                         in_data=orig_i)
            #
            # compare the field contents
            #
            self.assertTrue (np.all (np.equal (orig_o, o_field)))
            self.assertTrue (np.all (np.equal (orig_i, i_field)))

