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

from nose.plugins.attrib import attr

from gridtools.stencil import MultiStageStencil
from tests.test_stencils import CopyTest




class ShallowWater2D (MultiStageStencil):
    """
    Implements the shallow water equation as a multi-stage stencil.-
    """
    def __init__ (self, domain):
        """
        A comment to make AST parsing more difficult.-
        """
        super ( ).__init__ ( )

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
        # timestep
        #
        self.dt = 0.00125
        self.bl = 0.2

        #
        # temporary data fields
        #
        self.Hd = np.zeros (self.domain)
        self.Ud = np.zeros (self.domain)
        self.Vd = np.zeros (self.domain)

        self.Hx = np.zeros (self.domain)
        self.Ux = np.zeros (self.domain)
        self.Vx = np.zeros (self.domain)

        self.Hy = np.zeros (self.domain)
        self.Uy = np.zeros (self.domain)
        self.Vy = np.zeros (self.domain)

        self.Havg = np.zeros (self.domain)
        self.Uavg = np.zeros (self.domain)
        self.Vavg = np.zeros (self.domain)


    def droplet (self, H):
        """
        A two-dimensional falling drop into the water:

            H   the water height field.-
        """
        x,y = np.mgrid[:self.domain[0], :self.domain[1]]
        droplet_x, droplet_y = self.domain[0]/2, self.domain[1]/2
        rr = (x-droplet_x)**2 + (y-droplet_y)**2
        H[rr<(self.domain[0]/10)**2] = 1.1 # add a perturbation in pressure surface
        #x = np.array ([np.arange (-1, 1 + 2/(width-1), 2/(width-1))] * (width-1))
        #y = np.copy (x)
        #drop = height * np.exp (-5*(x*x + y*y))
        ##
        ## pad the resulting array with zeros
        ##
        ##zeros = np.zeros (shape[:-1])
        ##zeros[:drop.shape[0], :drop.shape[1]] = drop
        ##return zeros.reshape (zeros.shape[0],
        ##                      zeros.shape[1], 
        ##                      1)
        #return drop.reshape ((drop.shape[0],
        #                      drop.shape[1],
        #                      1))


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
        for p in self.get_interior_points (out_U):
            #
            # temporaries used later
            #
            self.Ux[p]   = out_U[p + (1,0,0)] - out_U[p + (-1,0,0)]
            self.Uy[p]   = out_U[p + (0,1,0)] - out_U[p + (0,-1,0)]
            self.Uavg[p] = (out_U[p + (1,0,0)] + out_U[p + (-1,0,0)] + out_U[p + (0,1,0)] + out_U[p + (0,-1,0)]) / 4.0
            #
            # diffusion
            #
            out_U[p] = out_U[p] * (1-self.bl) + self.bl * self.Uavg[p]
            #
            # dynamics
            #
            self.Ud[p] = -out_U[p] * self.Ux[p]

        for p in self.get_interior_points (out_V):
            #
            # temporaries used later
            #
            self.Vx[p]   = out_V[p + (1,0,0)] - out_V[p + (-1,0,0)]
            self.Vy[p]   = out_V[p + (0,1,0)] - out_V[p + (0,-1,0)]
            self.Vavg[p] = (out_V[p + (1,0,0)] + out_V[p + (-1,0,0)] + out_V[p + (0,1,0)] + out_V[p + (0,-1,0)]) / 4.0
            #
            # diffusion
            #
            out_V[p] = out_V[p] * (1-self.bl) + self.bl * self.Vavg[p]
            #
            # dynamics
            #
            self.Vd[p] = -out_V[p] * self.Vy[p]

        for p in self.get_interior_points (out_H):
            #
            # temporaries used later
            #
            self.Hx[p]   = out_H[p + (1,0,0)] - out_H[p + (-1,0,0)]
            self.Hy[p]   = out_H[p + (0,1,0)] - out_H[p + (0,-1,0)]
            self.Havg[p] = (out_H[p + (1,0,0)] + out_H[p + (-1,0,0)] + out_H[p + (0,1,0)] + out_H[p + (0,-1,0)]) / 4.0
            #
            # diffusion
            #
            out_H[p] = out_H[p] * (1-self.bl) + self.bl * self.Havg[p]
            #
            # dynamics
            #
            self.Ud[p] = self.Ud[p] - 0.12 * self.Hx[p]
            self.Vd[p] = self.Vd[p] - 0.12 * self.Hy[p]
            self.Hd[p] = self.Hd[p] - out_H[p] * (self.Ux[p] + self.Vy[p])
            #
            # take first-order Euler step
            #
            out_U[p] = out_U[p] + self.dt * self.Ud[p];
            out_V[p] = out_V[p] + self.dt * self.Vd[p];
            out_H[p] = out_H[p] + self.dt * self.Hd[p];




    def kernel_old (self, out_H, out_U, out_V):
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
            self.Hy[p] -= ( out_V[p + (1,1,0)] - out_V[p + (1,0,0)] ) * ( self.dt / (2*self.dy) )
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
            out_H[p] -= (self.dt / self.dx) * ( self.Ux[p + (0,-1,0)] - self.Ux[p + (-1,-1,0)] )
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



class ShallowWater2DTest (CopyTest):
    """
    A test case for the shallow water stencil defined above.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.DEBUG)

        self.domain = (16, 16, 1)

        self.params = ('out_H', 'out_U', 'out_V')
        self.temps  = ('Havg', 'Hd', 'Hx', 'Hy', 
                       'Uavg', 'Ud', 'Ux', 'Uy', 
                       'Vavg', 'Vd', 'Vx', 'Vy')

        self.out_H = np.ones  (self.domain)
        self.out_U = np.zeros (self.domain)
        self.out_V = np.zeros (self.domain)

        self.stencil = ShallowWater2D (self.domain)
        
        self.stencil.set_halo        ( (2, 2, 2, 2) )
        self.stencil.set_k_direction ("forward")


    @attr(lang='c++')
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
        #self.stencil.backend = 'c++'
        self.stencil.backend = 'python'

        #
        # disturb the water surface
        #
        self.stencil.droplet (self.out_H)

        #
        # show its evolution
        #
        for i in range (1000):
            #self.stencil.reflect_borders (self.out_H,
            #                            self.out_U,
            #                            self.out_V)
            self.stencil.run (out_H=self.out_H,
                              out_U=self.out_U,
                              out_V=self.out_V)

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
            #print ("%d - %s - sum(H): %s" % (framenumber,
            #                                 swobj.backend,
            #                                 np.sum (self.out_H)))
            #
            # reset if the system becomes unstable
            #
            if np.any (np.isnan (self.out_H)):
                self.setUp ( )
                self.stencil.droplet (self.out_H)
                print ("Reseting ...")

            ax.cla ( )
            surf = ax.plot_wireframe (X, Y, 
                                np.squeeze (self.out_H, axis=(2,)),
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


    @attr (lang='python')
    def test_python_execution (self):
        """
        Checks that the stencil results are correct if executing in Python mode.-
        """
        #self.stencil.reflect_borders (self.out_H,
        #                            self.out_U,
        #                            self.out_V)
        self.stencil.droplet (self.out_H)

        self.stencil.run (out_H=self.out_H,
                          out_U=self.out_U,
                          out_V=self.out_V)

        print (self.out_H[:,:,0])
        print (self.out_U[:,:,0])
        print (self.out_V[:,:,0])

        self.assertIsNotNone (self.out_H)
        self.assertIsNotNone (self.out_U)
        self.assertIsNotNone (self.out_V)

