##
## --- SHALLOW_WATER ---
##
## based on the MATLAB version of WAVE -- 2D Shallow Water Model
## New Mexico Supercomputing Challenge, Glorieta Kickoff 2007
##
## Lax-Wendroff finite difference method.
## Reflective boundary conditions.
## Random water drops initiate gravity waves.
## Surface plot displays height colored by momentum.
## Plot title shows t = simulated time and tv = a measure of total variation.
## An exact solution to the conservation law would have constant tv.
## Lax-Wendroff produces nonphysical oscillations and increasing tv.
##
## Cleve Moler, The MathWorks, Inc.
## Derived from C programs by
##    Bob Robey, Los Alamos National Laboratory.
##    Joseph Koby, Sarah Armstrong, Juan-Antonio Vigil, Vanessa Trujillo, McCurdy School.
##    Jonathan Robey, Dov Shlachter, Los Alamos High School.
## See:
##    http://en.wikipedia.org/wiki/Shallow_water_equations
##    http://www.amath.washington.edu/~rjl/research/tsunamis
##    http://www.amath.washington.edu/~dgeorge/tsunamimodeling.html
##    http://www.amath.washington.edu/~claw/applications/shallow/www
##
import unittest
import logging
import numpy as np

from numpy import zeros

from gridtools import MultiStageStencil



class ShallowWater (MultiStageStencil):
    """
    Implements the shallow water equation as a multi-stage stencil.-
    """
    def __init__ (self, domain):
        """
        A comment to make AST parsing more difficult.-
        """
        super (ShallowWater, self).__init__ ( )

        self.domain = domain
        #
        # grid size with a halo of one
        #
        self.n = domain[0] - 2

        #
        # step discretization step in (i, j) direction
        #
        self.dx = 1.0
        self.dy = 1.0

        #
        # gravity-accelleration constant
        #
        self.g = 9.8

        #
        # time and timestep
        #
        self.t  = 0.00
        self.dt = 0.02

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
        drop = self.droplet (2, 3)
        w = drop.shape[0]

        rand0 = np.random.rand ( )
        rand1 = np.random.rand ( )
        rand2 = np.random.rand ( )

        for i in range (w):
            i_idx = i + np.ceil (rand0 * (self.n - w))
            for j in range (w):
                j_idx = j + np.ceil (rand1 * (self.n - w))
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
        for p in self.get_interior_points (out_H):
            # height
            self.Hx[p]  = ( out_H[p + (1,1,0)] + out_H[p + (0,1,0)] ) / 2.0
            self.Hx[p] -= ( out_U[p + (1,1,0)] - out_U[p + (0,1,0)] ) * ( self.dt / (2*self.dx) )

            # X momentum    
            self.Ux[p]  = ( out_U[p + (1,1,0)] + out_U[p + (0,1,0)] ) / 2.0
            self.Ux[p] -=  ( ( (out_U[p + (1,1,0)]*out_U[p + (1,1,0)]) / out_H[p + (1,1,0)] + 
                               (out_H[p + (1,1,0)]*out_H[p + (1,1,0)]) * self.g / 2.0 ) -
                             ( (out_U[p + (0,1,0)]*out_U[p + (0,1,0)]) / out_H[p + (0,1,0)] + 
                               (out_H[p + (0,1,0)]*out_H[p + (0,1,0)]) * self.g / 2.0 )
                           ) * ( self.dt / (2*self.dx) )

            # Y momentum
            self.Vx[p]  = ( out_V[p + (1,1,0)] + out_V[p + (0,1,0)] ) / 2.0
            self.Vx[p] -= ( ( out_U[p + (1,1,0)] * out_V[p + (1,1,0)] / out_H[p + (1,1,0)] ) -
                            ( out_U[p + (0,1,0)] * out_V[p + (0,1,0)] / out_H[p + (0,1,0)] )
                          ) * ( self.dt / (2*self.dx) )

        #
        # first half step (stage Y direction)
        #
        for p in self.get_interior_points (out_H):
            # height
            self.Hy[p]  = ( out_H[p + (1,1,0)] + out_H[p + (1,0,0)] ) / 2.0
            self.Hy[p] -= self.dt / (2*self.dy) * ( out_V[p + (1,1,0)] - out_V[p+ (1,0,0)] )

            # X momentum
            self.Uy[p]  = ( out_U[p + (1,1,0)] + out_U[p + (1,0,0)] ) / 2.0
            self.Uy[p] -= self.dt / (2*self.dy) * ( ( out_V[p + (1,1,0)] * out_U[p + (1,1,0)] / out_H[p + (1,1,0)] ) -
                                                    ( out_V[p + (1,0,0)] * out_U[p + (1,0,0)] / out_H[p + (1,0,0)] )
                                                  )

            # Y momentum
            self.Vy[p]  = ( out_V[p + (1,1,0)] + out_V[p + (1,0,0)] ) / 2.0
            self.Vy[p] -= self.dt / (2*self.dy) * ( ( (out_V[p + (1,1,0)]*out_V[p + (1,1,0)]) / out_H[p + (1,1,0)] + 
                                                       self.g / 2 * (out_H[p + (1,1,0)]*out_H[p + (1,1,0)]) ) -
                                                    ( (out_V[p + (1,0,0)]*out_V[p + (1,0,0)]) / out_H[p + (1,0,0)] + 
                                                       self.g / 2 * (out_H[p + (1,0,0)]*out_H[p + (1,0,0)]) )
                                                  )

        #
        # second half step (stage)
        #
        for p in self.get_interior_points (out_H):
            # height
            out_H[p] -= (self.dt / self.dx) * ( self.Ux[p + (0,-2,0)] - self.Ux[p + (-1,-1,0)] )
            out_H[p] -= (self.dt / self.dy) * ( self.Vy[p + (-1,0,0)] - self.Vy[p + (-1,-1,0)] )

            # X momentum
            out_U[p] -= (self.dt / self.dx) * ( ( (self.Ux[p + (0,-1,0)]*self.Ux[p + (0,-1,0)]) / self.Hx[p + (0,-1,0)] + 
                                                   self.g / 2 * (self.Hx[p + (0,-1,0)]*self.Hx[p + (0,-1,0)]) ) -
                                                ( (self.Ux[p + (-1,-1,0)]*self.Ux[p + (-1,-1,0)]) / self.Hx[p + (-1,-1,0)] + 
                                                   self.g / 2 * (self.Hx[p + (-1,-1,0)]*self.Hx[p + (-1,-1,0)]) )
                                              )
            out_U[p] -= (self.dt / self.dy) * ( ( self.Vy[p + (-1,0,0)] * self.Uy[p + (-1,0,0)] / self.Hy[p + (-1,0,0)] ) - 
                                                 ( self.Vy[p + (-1,-1,0)] * self.Uy[p + (-1,-1,0)] / self.Hy[p + (-1,-1,0)] )
                                               )
            # Y momentum
            out_V[p] -= (self.dt / self.dx) * ( ( self.Ux[p + (0,-1,0)] * self.Vx[p + (0,-1,0)] / self.Hx[p + (0,-1,0)] ) -
                                                 ( self.Ux[p + (-1,-1,0)] * self.Vx[p + (-1,-1,0)] / self.Hx[p + (-1,-1,0)] )
                                               )
            out_V[p] -= (self.dt / self.dy) * ( ( (self.Vy[p + (-1,0,0)]*self.Vy[p + (-1,0,0)]) / self.Hy[p + (-1,0,0)] + 
                                                   self.g / 2 * (self.Hy[p + (-1,0,0)]*self.Hy[p + (-1,0,0)]) ) -
                                                ( (self.Vy[p + (-1,-1,0)]*self.Vy[p + (-1,-1,0)]) / self.Hy[p + (-1,-1,0)] + 
                                                   self.g / 2 * (self.Hy[p + (-1,-1,0)]*self.Hy[p + (-1,-1,0)]) )
                                              )




#class ShallowWaterTest (CopyTest):
class ShallowWaterTest (unittest.TestCase):
    """
    A test case for the shallow water stencil defined above.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.DEBUG)

        self.domain = (32, 32, 1)

        self.params = ('out_H', 'out_U', 'out_V')
        self.temps  = ('Hx', 'Hy', 'Ux', 'Uy', 'Vx', 'Vy')

        self.out_H = np.ones  (self.domain)
        self.out_U = np.zeros (self.domain)
        self.out_V = np.zeros (self.domain)

        self.stencil = ShallowWater (self.domain)
        self.stencil.set_halo        ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_interactive_plot (self):
        """
        Displays an animation inside a matplotlib graph.-
        """
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
        self.stencil.create_random_drop (self.H)

        #
        # show its evolution
        #
        for i in range (100):
            self.stencil.reflect_borders (self.H,
                                        self.U,
                                        self.V)
            self.stencil.run (out_H=self.H,
                            out_U=self.U,
                            out_V=self.V)
            print ("%d - %s - sum(H): %s" % (i,
                                             self.stencil.backend,
                                             np.sum (self.H)))

        """
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
            #print ("%d - %s - sum(H): %s" % (framenumber,
            #                                 swobj.backend,
            #                                 np.sum (self.H)))
            #
            # reset if the system becomes unstable
            #
            if np.any (np.isnan (self.H)):
                self.setUp ( )
                self.stencil.create_random_drop (self.H)
                print ("Reseting ...")

            ax.cla ( )
            surf = ax.plot_wireframe (X, Y, 
                                np.squeeze (self.H, axis=(2,)),
                                rstride=1, 
                                cstride=1, 
                                cmap=cm.jet, 
                                linewidth=1, 
                                antialiased=False) 
            #plt.savefig ("/tmp/water_%04d" % framenumber)
            return surf,

        anim = animation.FuncAnimation (fig,
                                        draw_frame,
                                        fargs=(self.stencil,),
                                        frames=range (100),
                                        interval=50,
                                        blit=False)
        plt.show ( )


    def test_automatic_range_detection (self):
        self.stencil.backend = 'c++'
        self.stencil.run ( )

        scope = self.stencil.inspector.functors[0].scope
       
        for p in self.params:
            self.assertEqual (scope[p].range, [0, 1, 0, 1])
    """


    def test_compare_python_and_native_executions (self):
        import copy

        water_py          = self.stencil
        water_cxx         = copy.deepcopy (self.stencil)
        water_cxx.backend = 'c++'

        #
        # disturb the water surface
        #
        self.stencil.create_random_drop (self.out_H)

        #
        # show its evolution
        #
        for i in range (100):
            self.stencil.reflect_borders (self.out_H,
                                          self.out_U,
                                          self.out_V)
            #
            # original content of the data fields
            #
            orig_H = np.array (self.out_H)
            orig_U = np.array (self.out_U)
            orig_V = np.array (self.out_V)

            #
            # apply the Python version of the stencil
            #
            water_py.run (out_H=self.out_H,
                          out_U=self.out_U,
                          out_V=self.out_V)
            #
            # apply the native version of the stencil
            #
            water_cxx.run (out_H=orig_H,
                           out_U=orig_U,
                           out_V=orig_V)
            #
            # compare the field contents
            #
            print ('%d - H - %s == %s' % (i, np.sum (orig_H), np.sum (self.out_H)))
            #self.assertTrue (np.all (np.equal (orig_H, self.out_H)))
            print ('%d - U - %s == %s' % (i, np.sum (orig_U), np.sum (self.out_U)))
            #self.assertTrue (np.all (np.equal (orig_U, self.out_U)))
            print ('%d - V - %s == %s' % (i, np.sum (orig_V), np.sum (self.out_V)))
            #self.assertTrue (np.all (np.equal (orig_V, self.out_V)))


    """
    def test_symbol_discovery (self):
        ""
        Checks that all the symbols have been correctly recognized.-
        ""
        self.stencil.backend = 'c++'
        self.stencil.run (out_H=self.H,
                        out_U=self.U,
                        out_V=self.V)
        #
        # check input/output fields were correctly discovered
        #
        insp = self.stencil.inspector
        out_fields = ['out_H', 'out_U', 'out_V']
        for f in out_fields:
            self.assertIsNotNone (insp.symbols[f])
            self.assertTrue (insp.symbols.is_parameter (f))

        #
        # check temporary fields were correctly discovered
        #
        tmp_fields = [
        for f in tmp_fields:
            self.assertIsNotNone (insp.symbols[f])
            self.assertTrue (insp.symbols.is_temporary (f))
        

    def test_python_execution (self):
        ""
        Checks that the stencil results are correct if executing in Python mode.-
        ""
        self.stencil.reflect_borders (self.H,
                                    self.U,
                                    self.V)
        self.stencil.run (out_H=self.H,
                        out_U=self.U,
                        out_V=self.V)
        self.assertIsNotNone (self.H)
        self.assertIsNotNone (self.U)
        self.assertIsNotNone (self.V)
     """

