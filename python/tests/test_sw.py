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

from gridtools import MultiStageStencil, StencilInspector



class ShallowWater (MultiStageStencil):
    """
    Implements the shallow water equation as a multi-stage stencil.-
    """
    def __init__ (self):
        """
        A comment to make AST parsing more difficult.-
        """
        super (ShallowWater, self).__init__ ( )
        #
        # grid size
        #
        self.n = 64
        #self.n = 14

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


    def droplet (self, height, width, shape):
        """
        A two-dimensional Gaussian of the falling drop into the water:

            height  height of the generated drop;
            width   width of the generated drop;
            shape   shape of the resulting array, padded with zeros.-
        """
        x = np.array ([np.arange (-1, 1 + 2/(width-1), 2/(width-1))] * (width-1))
        y = np.copy (x)
        drop = height * np.exp (-5*(x*x + y*y))
        #
        # pad the resulting array with zeros
        #
        zeros = np.zeros (shape[:-1])
        zeros[:drop.shape[0], :drop.shape[1]] = drop
        return zeros.reshape (zeros.shape[0],
                              zeros.shape[1], 
                              1)

    def create_random_drop (self, H, domain):
        """
        Disturbs the water surface with a drop.-
        """
        drop = self.droplet (5, 11, domain)
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



    def kernel (self, out_H, out_U, out_V, in_drop):
        """
        This stencil comprises multiple stages.-
        """
        #
        # first half step (stage X direction)
        #
        for p in self.get_interior_points (self.Hx,
                                           k_direction="forward"):

            # height
            self.Hx[p]  = ( out_H[p + (1,1,0)] + out_H[p + (0,1,0)] ) / 2.0
            self.Hx[p] -= self.dt / (2*self.dx) * ( out_U[p + (1,1,0)] - out_U[p + (0,1,0)] )

            # X momentum    
            self.Ux[p]  = ( out_U[p + (1,1,0)] + out_U[p + (0,1,0)] ) / 2.0
            self.Ux[p] -= self.dt / (2*self.dx) * ( ( (out_U[p + (1,1,0)]*out_U[p + (1,1,0)]) / out_H[p + (1,1,0)] + 
                                                       self.g / 2.0 * (out_H[p + (1,1,0)]*out_H[p + (1,1,0)]) ) -
                                                    ( (out_U[p + (0,1,0)]*out_U[p + (0,1,0)]) / out_H[p + (0,1,0)] + 
                                                       self.g / 2.0 * (out_H[p + (0,1,0)]*out_H[p + (0,1,0)]) )
                                                  )
            # Y momentum
            self.Vx[p]  = ( out_V[p + (1,1,0)] + out_V[p + (0,1,0)] ) / 2.0
            self.Vx[p] -= self.dt / (2*self.dx) * ( ( out_U[p + (1,1,0)] * out_V[p + (1,1,0)] / out_H[p + (1,1,0)] ) -
                                                    ( out_U[p + (0,1,0)] * out_V[p + (0,1,0)] / out_H[p + (0,1,0)] )
                                                  )
        #
        # first half step (stage Y direction)
        #
        for p in self.get_interior_points (self.Hy,
                                           k_direction="forward"):
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
        for p in self.get_interior_points (self.Hx,
                                           k_direction="forward"):
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





class ShallowWaterTest (unittest.TestCase):
    """
    A test case for the shallow water stencil defined above.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.INFO)

        self.domain = (66, 66, 1)
        #self.domain = (16, 16, 1)

        self.H = np.ones  (self.domain)
        self.U = np.zeros (self.domain)
        self.V = np.zeros (self.domain)

        self.water = ShallowWater ( )
        self.drop  = np.ones (self.domain)


    def test_interactive_plot (self):
        import matplotlib.pyplot as plt
        from matplotlib import animation, cm
        from mpl_toolkits.mplot3d import Axes3D

        #
        # enable native execution for the stencil
        #
        #self.water.backend = 'c++'

        #
        # disturb the water surface
        #
        self.water.create_random_drop (self.H,
                                       self.domain)

        """
        for i in range (5):
            self.water.reflect_borders (self.H,
                                        self.U,
                                        self.V)
            self.water.run (out_H=self.H,
                            out_U=self.U,
                            out_V=self.V,
                            in_drop=self.drop)
            print (self.H)
            input ("Press Enter to continue ...")
        """

        #
        # initialize 3D plot
        #
        fig = plt.figure ( )
        ax  = fig.add_subplot (111, 
                               projection='3d')

        rng = np.arange (self.domain[0])
        X, Y = np.meshgrid (rng, rng)
        surf = ax.plot_surface (X, Y, 
                                np.squeeze (self.H, axis=(2,)),
                                rstride=1, 
                                cstride=1, 
                                cmap=cm.jet, 
                                linewidth=0, 
                                antialiased=False) 
        fig.show ( )
         
        #
        # animation update function
        #
        def draw_frame (framenumber, swobj):
            swobj.reflect_borders (self.H,
                                   self.U,
                                   self.V)
            swobj.run (out_H=self.H,
                       out_U=self.U,
                       out_V=self.V,
                       in_drop=self.drop)
            #
            # reset if the system becomes unstable
            #
            if np.any (np.isnan (self.H)):
                self.setUp ( )
                self.water.create_random_drop (self.H,
                                               self.domain)
                input ("Reseting ...")

            ax.clear ( )
            surf = ax.plot_surface (X, Y, 
                                np.squeeze (self.H, axis=(2,)),
                                rstride=1, 
                                cstride=1, 
                                cmap=cm.jet, 
                                linewidth=0, 
                                antialiased=False) 
            #plt.savefig ("/tmp/water_%04d" % framenumber)
            return surf,

        plt.ion ( )
        anim = animation.FuncAnimation (fig,
                                        draw_frame,
                                        fargs=(self.water,),
                                        interval=20,
                                        blit=False)


    def test_symbol_discovery (self):
        """
        Checks that all the symbols have been correctly recognized.-
        """
        self.water.backend = 'c++'
        self.water.run (out_H=self.H,
                        out_U=self.U,
                        out_V=self.V,
                        in_drop=self.drop)
        #
        # check input/output fields were correctly discovered
        #
        insp = self.water.inspector
        out_fields = ['out_H', 'out_U', 'out_V', 'in_drop']
        for f in out_fields:
            self.assertIsNotNone (insp.symbols[f])
            self.assertTrue (insp.symbols.is_parameter (f))
            self.assertTrue (insp.symbols.is_parameter (f, 'kernel'))

        #
        # check temporary fields were correctly discovered
        #
        tmp_fields = ['Hx', 'Hy', 'Ux', 'Uy', 'Vx', 'Vy']
        for f in tmp_fields:
            self.assertIsNotNone (insp.symbols[f])
            self.assertTrue (insp.symbols.is_temporary (f))
        

    def test_python_execution (self):
        """
        Checks that the stencil results are correct if executing in Python mode.-
        """
        self.water.reflect_borders (self.H,
                                    self.U,
                                    self.V)
        self.water.run (out_H=self.H,
                        out_U=self.U,
                        out_V=self.V,
                        in_drop=self.drop)
        self.assertIsNotNone (self.H)
        self.assertIsNotNone (self.U)
        self.assertIsNotNone (self.V)
        self.assertIsNotNone (self.drop)


    def test_native_execution (self):
        """
        Checks that the stencil results are correct if executing in native mode.-
        """
        self.water.backend = 'c++'
        self.water.run (out_H=self.H,
                        out_U=self.U,
                        out_V=self.V,
                        in_drop=self.drop)
        #
        # the library object should contain a valid reference
        # if compilation was successful
        #
        self.assertIsNotNone (self.water.inspector.lib_obj)





"""
class ShallowWaterEquation (object):

    def step (self, nstep): 
 
        # random water drops
        if nstep == 0:
            w = self.drop.shape[0]

            rand0 = np.random.rand ( )
            rand1 = np.random.rand ( )
            rand2 = np.random.rand ( )

            for i in range (w):
                i_idx = i + np.ceil (rand0 * (self.n - w))
                for j in range (w):
                    j_idx = j + np.ceil (rand1 * (self.n - w))
                    H[i_idx, j_idx] += rand2 * self.drop[i, j]

        # reflective boundary conditions
        H[:,0] = H [:,1];      U [:,0] = U [:,1];      V [:,0] = -V[:,1]
        H[:,n+1] = H [:,n+0];  U [:,n+1] = U [:,n+0];  V [:,n+1] = -V[:,n+0]
        H[0,:] = H [1,:];      U [0,:] = -U[1,:];      V [0,:] = V [1,:]
        H[n+1,:] = H [n+0,:];  U [n+1,:] = -U[n+0,:];  V [n+1,:] = V [n+0,:]

        # first half step (stage X direction)
        for i in range (n + 1):
            for j in range (n):
                # height
                Hx[i, j]  = ( H[i+1, j+1] + H[i, j+1] ) / 2
                Hx[i, j] -= dt / (2*dx) * ( U[i+1, j+1] - U[i, j+1] )

                # X momentum    
                Ux[i, j]  = ( U[i+1, j+1] + U[i, j+1] ) / 2
                Ux[i, j] -= dt / (2*dx) * ( ( U[i+1, j+1]**2 / H[i+1, j+1] + g/2*H[i+1, j+1]**2 ) -
                                            ( U[i, j+1]**2 / H[i, j+1] + g/2*H[i, j+1]**2 )
                                          )
                # Y momentum
                Vx[i, j]  = ( V[i+1, j+1] + V[i, j+1] ) / 2 
                Vx[i, j] -= dt / (2*dx) * ( ( U[i+1, j+1] * V[i+1, j+1] / H[i+1, j+1] ) -
                                            ( U[i, j+1] * V[i, j+1] / H[i, j+1] )
                                          )
        # first half step (stage Y direction)
        for i in range (n):
            for j in range (n + 1):
                # height
                Hy[i, j]  = ( H[i+1, j+1] + H[i+1, j] ) / 2
                Hy[i, j] -= dt / (2*dy) * ( V[i+1, j+1] - V[i+1, j] )

                # X momentum
                Uy[i, j]  = ( U[i+1, j+1] + U[i+1, j] ) / 2 
                Uy[i, j] -= dt / (2*dy) * ( ( V[i+1, j+1] * U[i+1, j+1] / H[i+1, j+1] ) -
                                            ( V[i+1, j] * U[i+1, j] / H[i+1, j] )
                                          )
                # Y momentum
                Vy[i, j]  = ( V[i+1, j+1] + V[i+1, j] ) / 2
                Vy[i, j] -= dt / (2*dy) * ( ( V[i+1, j+1]**2 / H[i+1, j+1] + g/2*H[i+1, j+1]**2 ) -
                                            ( V[i+1, j]**2 / H[i+1, j] + g/2*H[i+1, j]**2 )
                                          )

        # second half step (stage)
        for i in range (1, n + 1):
            for j in range (1, n + 1):
                # height
                H[i, j] -= (dt/dx) * ( Ux[i, j-1] - Ux[i-1, j-1] )
                H[i, j] -= (dt/dy) * ( Vy[i-1, j] - Vy[i-1, j-1] )

                # X momentum
                U[i, j] -= (dt/dx) * ( ( Ux[i, j-1]**2 / Hx[i, j-1] + g/2*Hx[i, j-1]**2 ) -
                                       ( Ux[i-1, j-1]**2 / Hx[i-1, j-1] + g/2*Hx[i-1, j-1]**2 )
                                     )
                U[i, j] -= (dt/dy) * ( ( Vy[i-1, j] * Uy[i-1, j] / Hy[i-1, j] ) - 
                                       ( Vy[i-1, j-1] * Uy[i-1, j-1] / Hy[i-1, j-1] )
                                     )
                # Y momentum
                V[i, j] -= (dt/dx) * ( ( Ux[i, j-1] * Vx[i, j-1] / Hx[i, j-1] ) -
                                       ( Ux[i-1, j-1] * Vx[i-1, j-1] / Hx[i-1, j-1] )
                                     )
                V[i, j] -= (dt/dy) * ( ( Vy[i-1, j]**2 / Hy[i-1, j] + g/2*Hy[i-1, j]**2 ) -
                                       ( Vy[i-1, j-1]**2 / Hy[i-1, j-1] + g/2*Hy[i-1, j-1]**2 )
                                     )

        # reset if the system becomes unstable
        #if all (all (isnan (H))), break, end  % Unstable, restart
        print ("%d " % nstep)






# the shallow water stencil object
#
sw = ShallowWater ( )

"""

