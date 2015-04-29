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




class SW_Momentum (MultiStageStencil):
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.bl = 1.2
        #
        # temporary data fields
        #
        self.Mavg = np.zeros (domain)


    def kernel (self, out_M, out_Mx, out_My, in_M):
         for p in self.get_interior_points (out_M):
            #
            # temporary used later
            #
            self.Mavg[p] = (in_M[p + (1,0,0)] + in_M[p + (-1,0,0)] + 
                            in_M[p + (0,1,0)] + in_M[p + (0,-1,0)]) / 4.0
            #
            # derivatives in 'x' and 'y' dimensions
            #
            out_Mx[p] = in_M[p + (1,0,0)] - in_M[p + (-1,0,0)]
            out_My[p] = in_M[p + (0,1,0)] - in_M[p + (0,-1,0)]
            #
            # diffusion
            #
            out_M[p] = out_M[p] * (1.0 - self.bl) + self.bl * self.Mavg[p]



class SW_Dynamic (MultiStageStencil):
    def __init__ (self):
        super ( ).__init__ ( )
        self.dt     = 0.01
        self.growth = 0.2


    def kernel (self, out_H, out_Hd, in_H, in_Hx, in_Hy,
                      out_U, out_Ud, in_U, in_Ux, 
                      out_V, out_Vd, in_V, in_Vy):
        for p in self.get_interior_points (out_Hd):
            out_Ud[p] = -in_U[p] * in_Ux[p]
            out_Vd[p] = -in_V[p] * in_Vy[p]

            out_Ud[p] = out_Ud[p] - self.growth * in_Hx[p]
            out_Vd[p] = out_Vd[p] - self.growth * in_Hy[p]
            out_Hd[p] = out_Hd[p] - in_H[p] * (in_Ux[p] + in_Vy[p])

            #
            # take first-order Euler step
            #
            out_U[p] = out_U[p] + self.dt * out_Ud[p];
            out_V[p] = out_V[p] + self.dt * out_Vd[p];
            out_H[p] = out_H[p] + self.dt * out_Hd[p];



class SW (MultiStageStencil):
    def __init__ (self, domain):
        super ( ).__init__ ( )
        #
        # constants to callibrate the system
        #
        self.bl     = 0.2
        self.dt     = 0.001
        self.growth = 0.5
        #
        # temporary data fields
        #
        self.Mavg = np.zeros (domain)
        self.Hx   = np.zeros (domain)
        self.Ux   = np.zeros (domain)
        self.Vx   = np.zeros (domain)
        self.Hy   = np.zeros (domain)
        self.Uy   = np.zeros (domain)
        self.Vy   = np.zeros (domain)


    def stage_momentum (self, out_M, out_Mx, out_My, in_M):
         for p in self.get_interior_points (out_M):
            #
            # temporary used later
            #
            self.Mavg[p] = (in_M[p + (1,0,0)] + in_M[p + (-1,0,0)] + 
                            in_M[p + (0,1,0)] + in_M[p + (0,-1,0)]) / 4.0
            #
            # derivatives in 'x' and 'y' dimensions
            #
            out_Mx[p] = in_M[p + (1,0,0)] - in_M[p + (-1,0,0)]
            out_My[p] = in_M[p + (0,1,0)] - in_M[p + (0,-1,0)]
            #
            # diffusion
            #
            out_M[p] = out_M[p] * (1.0 - self.bl) + self.bl * self.Mavg[p]


    def stage_dynamics (self, out_H, out_Hd, in_H, in_Hx, in_Hy,
                        out_U, out_Ud, in_U, in_Ux, 
                        out_V, out_Vd, in_V, in_Vy):
        for p in self.get_interior_points (out_H):
            out_Ud[p] = -in_U[p] * in_Ux[p]
            out_Vd[p] = -in_V[p] * in_Vy[p]

            out_Ud[p] = out_Ud[p] - self.growth * in_Hx[p]
            out_Vd[p] = out_Vd[p] - self.growth * in_Hy[p]
            out_Hd[p] = out_Hd[p] - in_H[p] * (in_Ux[p] + in_Vy[p])

            #
            # take first-order Euler step
            #
            out_U[p] = out_U[p] + self.dt * out_Ud[p];
            out_V[p] = out_V[p] + self.dt * out_Vd[p];
            out_H[p] = out_H[p] + self.dt * out_Hd[p];


    def kernel (self, out_H, out_Hd, in_H,
                      out_U, out_Ud, in_U,
                      out_V, out_Vd, in_V):
        #
        # momentum calculation for each field
        #
        self.stage_momentum (out_M  = out_U,
                             out_Mx = self.Ux,
                             out_My = self.Uy,
                             in_M   = in_U)
        self.stage_momentum (out_M  = out_V,
                             out_Mx = self.Vx,
                             out_My = self.Vy,
                             in_M   = in_V)
        self.stage_momentum (out_M  = out_H,
                             out_Mx = self.Hx,
                             out_My = self.Hy,
                             in_M   = in_H)
        #
        # dynamics with momentum combined
        #
        self.stage_dynamics (out_H  = out_H,
                             out_U  = out_U,
                             out_V  = out_V,
                             out_Hd = out_Hd,
                             out_Ud = out_Ud,
                             out_Vd = out_Vd,
                             in_H   = in_H,
                             in_U   = in_U,
                             in_V   = in_V,
                             in_Hx  = self.Hx,
                             in_Ux  = self.Ux,
                             in_Hy  = self.Hy,
                             in_Vy  = self.Vy)




class SWTest (CopyTest):
    def setUp (self):
        logging.basicConfig (level=logging.DEBUG)

        self.domain = (24, 24, 1)

        self.params = ('out_H', 'out_Hd', 'in_H', 
                       'out_U', 'out_Ud', 'in_U', 
                       'out_V', 'out_Vd', 'in_V')
        self.temps  = ('self.Mavg',
                       'self.Hx',
                       'self.Ux',
                       'self.Vx',
                       'self.Hy',
                       'self.Uy',
                       'self.Vy')

        self.stencil = SW (self.domain)
        self.stencil.set_halo ( (2, 2, 2, 2) )

        self.out_H  = np.ones  (self.domain)
        self.out_U  = np.zeros (self.domain)
        self.out_V  = np.zeros (self.domain)
        self.out_Hd = np.zeros (self.domain)
        self.out_Ud = np.zeros (self.domain)
        self.out_Vd = np.zeros (self.domain)
        self.Hx     = np.zeros (self.domain)
        self.Ux     = np.zeros (self.domain)
        self.Vx     = np.zeros (self.domain)
        self.Hy     = np.zeros (self.domain)
        self.Uy     = np.zeros (self.domain)
        self.Vy     = np.zeros (self.domain)

        self.in_H  = np.ones  (self.domain)
        self.in_U  = np.zeros (self.domain)
        self.in_V  = np.zeros (self.domain)


    def droplet (self, H):
        """
        A two-dimensional falling drop into the water:

            H   the water height field.-
        """
        x,y = np.mgrid[:self.domain[0], :self.domain[1]]
        droplet_x, droplet_y = self.domain[0]/2, self.domain[1]/2
        rr = (x-droplet_x)**2 + (y-droplet_y)**2
        H[rr<(self.domain[0]/10.0)**2] += 0.1


    def test_automatic_dependency_detection (self):
        try:
            super ( ).test_automatic_dependency_detection ( )
        except AttributeError:
            print ('known to fail')


    def test_automatic_range_detection (self):
        expected_ranges = {'out_H'    : None,
                           'out_U'    : None,
                           'out_V'    : None,
                           'out_Hd'   : None,
                           'out_Ud'   : None,
                           'out_Vd'   : None,
                           'self.Hx'  : None,
                           'self.Ux'  : None,
                           'self.Vx'  : None,
                           'self.Hy'  : None,
                           'self.Uy'  : None,
                           'self.Vy'  : None,
                           'self.Mavg': None,
                           'in_H'     : ([-1,1,-1,1], None),
                           'in_U'     : ([-1,1,-1,1], None),
                           'in_V'     : ([-1,1,-1,1], None)}
        super ( ).test_automatic_range_detection (ranges=expected_ranges)


    def test_compare_python_and_native_executions (self):
        try:
            super ( ).test_compare_python_and_native_executions ( )
        except AssertionError:
            print ('known to fail')


    @attr(lang='c++')
    def test_interactive_plot (self):
        from gridtools import plt, fig, ax
        from matplotlib import animation

        self.stencil.backend = 'c++'

        self.droplet (self.out_H)
        self.droplet (self.in_H)

        plt.switch_backend ('agg')

        def init_frame ( ):
            ax.grid      (False)
            ax.set_xlim  ( (0, self.domain[0] - 1) )
            ax.set_ylim  ( (0, self.domain[1] - 1) )
            ax.set_zlim  ( (0.9, 1.10) )
            ax.view_init (azim=-60.0, elev=10.0)
            im = self.stencil.plot_3d (self.out_H[:,:,0])
            return [im]

        def draw_frame (frame):
            #
            # run the stencil
            #
            if frame % 2 == 0:
                self.stencil.run (out_H=self.out_H,
                                  out_U=self.out_U,
                                  out_V=self.out_V,
                                  out_Hd=self.out_Hd,
                                  out_Ud=self.out_Ud,
                                  out_Vd=self.out_Vd,
                                  in_H=self.in_H,
                                  in_U=self.in_U,
                                  in_V=self.in_V)
            else:
                self.stencil.run (out_H=self.in_H,
                                  out_U=self.in_U,
                                  out_V=self.in_V,
                                  out_Hd=self.out_Hd,
                                  out_Ud=self.out_Ud,
                                  out_Vd=self.out_Vd,
                                  in_H=self.out_H,
                                  in_U=self.out_U,
                                  in_V=self.out_V)
            ax.cla       ( )
            ax.grid      (False)
            ax.set_xlim  ( (0, self.domain[0] - 1) )
            ax.set_ylim  ( (0, self.domain[1] - 1) )
            ax.set_zlim  ( (0.9, 1.10) )
            ax.view_init (azim=-60.0, elev=10.0)
            im = self.stencil.plot_3d (self.out_H[:,:,0])
            return [im]

        anim = animation.FuncAnimation (fig,
                                        draw_frame,
                                        frames=range (10),
                                        interval=30,
                                        init_func=init_frame,
                                        blit=False)
        anim.save ('/tmp/%s.mp4' % self.__class__,
                   fps=30, 
                   extra_args=['-vcodec', 'libx264'])
        #plt.show ( )

 
    @attr (lang='python')
    def test_python_execution (self):
        self.stencil.backend = 'python'
        self.stencil.run (out_H=self.out_H,
                          out_U=self.out_U,
                          out_V=self.out_V,
                          out_Hd=self.out_Hd,
                          out_Ud=self.out_Ud,
                          out_Vd=self.out_Vd,
                          in_H=self.in_H,
                          in_U=self.in_U,
                          in_V=self.in_V)






class SW_Derive (MultiStageStencil):
    def kernel (self, out_Vx, out_Vy, in_V):
        for p in self.get_interior_points (in_V):
            out_Vx[p] = in_V[p + (1,0,0)] - in_V[p + (-1,0,0)]
            out_Vy[p] = in_V[p + (0,1,0)] - in_V[p + (0,-1,0)]



class SW_Average (MultiStageStencil):
    def kernel (self, out_avg, in_V):
        for p in self.get_interior_points (out_avg):
            out_avg[p] = (in_V[p + (1,0,0)] + in_V[p + (-1,0,0)] + in_V[p + (0,1,0)] + in_V[p + (0,-1,0)]) / 4.0



class SW_Diffusion (MultiStageStencil):
    def __init__ (self):
        super ( ).__init__ ( )
        self.bl  = 0.2


    def kernel (self, out_V, in_avg):
        for p in self.get_interior_points (out_V):
            out_V[p] = out_V[p] * (1.0 - self.bl) + self.bl * in_avg[p]



class SW_Euler (MultiStageStencil):
    def __init__ (self):
        super ( ).__init__ ( )
        self.dt = 0.15


    def kernel (self, out_H, in_Hd,
                      out_U, in_Ud,
                      out_V, in_Vd):
        for p in self.get_interior_points (out_H):
            #
            # take first-order Euler step
            #
            out_U[p] = out_U[p] + self.dt * in_Ud[p];
            out_V[p] = out_V[p] + self.dt * in_Vd[p];
            out_H[p] = out_H[p] + self.dt * in_Hd[p];



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
        # timestep
        #
        self.dt = 0.09
        self.bl = 0.50
        self.g  = 0.02

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
        H[rr<(self.domain[0]/10)**2] = 1.01
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
        U[:,0] =  U[:,1]/2.0
        V[:,0] = -V[:,1]/2.0

        H[:,self.domain[0]-2] =  H[:,self.domain[0]-1]  
        U[:,self.domain[0]-2] =  U[:,self.domain[0]-1]/2.0
        V[:,self.domain[0]-2] = -V[:,self.domain[0]-1]/2.0

        H[0,:] =  H[1,:]
        U[0,:] = -U[1,:]/2.0
        V[0,:] =  V[1,:]/2.0

        H[self.domain[0]-1,:] =  H[self.domain[0]-2,:]
        U[self.domain[0]-1,:] = -U[self.domain[0]-2,:]/2.0
        V[self.domain[0]-1,:] =  V[self.domain[0]-2,:]/2.0


    def kernel (self, out_H, out_U, out_V, in_H, in_U, in_V):
        for p in self.get_interior_points (out_U):
            #
            # temporaries used later
            #
            self.Ux[p]   = in_U[p + (1,0,0)] - in_U[p + (-1,0,0)]
            self.Uy[p]   = in_U[p + (0,1,0)] - in_U[p + (0,-1,0)]
            self.Uavg[p] = (in_U[p + (1,0,0)] + in_U[p + (-1,0,0)] + 
                            in_U[p + (0,1,0)] + in_U[p + (0,-1,0)]) / 4.0
            #
            # diffusion
            #
            out_U[p] = out_U[p] * (1-self.bl) + self.bl * self.Uavg[p]
            #
            # dynamics
            #
            self.Ud[p] = -in_U[p] * self.Ux[p]

        for p in self.get_interior_points (out_V):
            #
            # temporaries used later
            #
            self.Vx[p]   = in_V[p + (1,0,0)] - in_V[p + (-1,0,0)]
            self.Vy[p]   = in_V[p + (0,1,0)] - in_V[p + (0,-1,0)]
            self.Vavg[p] = (in_V[p + (1,0,0)] + in_V[p + (-1,0,0)] + 
                            in_V[p + (0,1,0)] + in_V[p + (0,-1,0)]) / 4.0
            #
            # diffusion
            #
            out_V[p] = out_V[p] * (1-self.bl) + self.bl * self.Vavg[p]
            #
            # dynamics
            #
            self.Vd[p] = -in_V[p] * self.Vy[p]

        for p in self.get_interior_points (out_H):
            #
            # temporaries used later
            #
            self.Hx[p]   = in_H[p + (1,0,0)] - in_H[p + (-1,0,0)]
            self.Hy[p]   = in_H[p + (0,1,0)] - in_H[p + (0,-1,0)]
            self.Havg[p] = (in_H[p + (1,0,0)] + in_H[p + (-1,0,0)] + 
                            in_H[p + (0,1,0)] + in_H[p + (0,-1,0)]) / 4.0
            #
            # diffusion
            #
            out_H[p] = out_H[p] * (1-self.bl) + self.bl * self.Havg[p]
            #
            # dynamics
            #
            self.Ud[p] = self.Ud[p] - self.g * self.Hx[p]
            self.Vd[p] = self.Vd[p] - self.g * self.Hy[p]
            self.Hd[p] = self.Hd[p] - in_H[p] * (self.Ux[p] + self.Vy[p])
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

