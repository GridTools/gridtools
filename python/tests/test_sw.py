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




class SW_U (MultiStageStencil):
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.bl = 0.2
        #
        # temporary data fields
        #
        self.Uavg  = np.zeros (domain)


    def kernel (self, out_U, out_Ud, out_Ux):
        for p in self.get_interior_points (out_U):
            #
            # temporaries used later
            #
            out_Ux[p]   = out_U[p + (1,0,0)] - out_U[p + (-1,0,0)]
            self.Uavg[p] = (out_U[p + (1,0,0)] + out_U[p + (-1,0,0)] + out_U[p + (0,1,0)] + out_U[p + (0,-1,0)]) / 4.0
            #
            # diffusion
            #
            out_U[p] = out_U[p] * (1.0 - self.bl) + self.bl * self.Uavg[p]
            #
            # dynamics
            #



class SW_UTest (CopyTest):
    """
    A test case for the shallow water stencil defined above.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.DEBUG)

        self.domain = (8, 8, 1)

        self.params = ('out_U', 'out_Ud', 'out_Ux')
        self.temps  = [ 'self.Uavg' ]

        self.out_U  = np.random.rand (*self.domain)
        self.out_Ud = np.random.rand (*self.domain)
        self.out_Ux = np.random.rand (*self.domain)

        self.stencil = SW_U (self.domain)
        
        self.stencil.set_halo        ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_automatic_dependency_detection (self):
        expected_deps = [('out_Ux',    'out_U'),
                         ('self.Uavg', 'out_U'),
                         ('out_U',     'out_U'),
                         ('out_U',     'self.Uavg'),
                         ('out_Ud',    'out_U'),
                         ('out_Ud',    'out_Ux')]
        super ( ).test_automatic_dependency_detection (deps=expected_deps)


    def test_automatic_range_detection (self):
        expected_ranges = {'out_U'    : [-1,1,-1,1],
                           'out_Ud'   : None,
                           'out_Ux'   : None}
        super ( ).test_automatic_range_detection (ranges=expected_ranges)


    @attr (lang='python')
    def test_python_execution (self):
        for i in range (2):
            self.stencil.run (out_U=self.out_U,
                              out_Ud=self.out_Ud,
                              out_Ux=self.out_Ux)



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



class SW_Dynamic (MultiStageStencil):
    def __init__ (self):
        super ( ).__init__ ( )
        self.growth = 1.2


    def kernel (self, out_Hd, in_H, in_Hx, in_Hy,
                      out_Ud, in_U, in_Ux, 
                      out_Vd, in_V, in_Vy):
        for p in self.get_interior_points (out_Hd):
            out_Ud[p] = -in_U[p] * in_Ux[p]
            out_Ud[p] = out_Ud[p] - self.growth * in_Hx[p]
            out_Vd[p] = -in_V[p] * in_Vy[p]
            out_Vd[p] = out_Vd[p] - self.growth * in_Hy[p]
            out_Hd[p] = out_Hd[p] - in_H[p] * (in_Ux[p] + in_Vy[p])


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



class SW_HTest (CopyTest):
    """
    A test case for the shallow water stencil defined above.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.DEBUG)

        self.domain = (8, 8, 1)

        self.params = ('out_H', 'out_Hd', 'in_Hx', 'in_Hy',
                       'out_U', 'out_Ud', 'in_Ux', 
                       'out_V', 'out_Vd', 'in_Vy')
        self.temps  = ( )

        self.out_H    = np.ones  (self.domain)
        self.out_Hd   = np.zeros (self.domain)
        self.out_U    = np.zeros (self.domain)
        self.out_Ud   = np.zeros (self.domain)
        self.out_V    = np.zeros (self.domain)
        self.out_Vd   = np.zeros (self.domain)

        self.in_Hx = np.zeros (self.domain)
        self.in_Hy = np.zeros (self.domain)
        self.in_Ux = np.zeros (self.domain)
        self.in_Uy = np.zeros (self.domain)
        self.in_Vx = np.zeros (self.domain)
        self.in_Vy = np.zeros (self.domain)

        self.avg_H = np.zeros (self.domain)
        self.avg_U = np.zeros (self.domain)
        self.avg_V = np.zeros (self.domain)


    def droplet (self, H):
        """
        A two-dimensional falling drop into the water:

            H   the water height field.-
        """
        x,y = np.mgrid[:self.domain[0], :self.domain[1]]
        droplet_x, droplet_y = self.domain[0]/2, self.domain[1]/2
        rr = (x-droplet_x)**2 + (y-droplet_y)**2
        H[rr<(self.domain[0]/50.0)**2] = 1.1 


    def test_automatic_range_detection (self):
        expected_ranges = {'out_H'    : [-1,1,-1,1],
                           'out_Hd'   : None,
                           'out_U'    : None,
                           'out_Ud'   : None,
                           'out_V'    : None,
                           'out_Vd'   : None,
                           'in_Ux'    : None,
                           'in_Vy'    : None}
        super ( ).test_automatic_range_detection (ranges=expected_ranges)


    @attr(lang='c++')
    def test_interactive_plot (self):
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d import axes3d

        self.der = SW_Derive    ( )
        self.avg = SW_Average   ( )
        self.dif = SW_Diffusion ( )
        self.dyn = SW_Dynamic   ( )
        self.eul = SW_Euler     ( )

        halo = (1, 1, 1, 1)
        self.der.set_halo ( halo )
        self.avg.set_halo ( halo )
        self.dif.set_halo ( halo )
        self.dyn.set_halo ( halo )
        self.eul.set_halo ( halo )

        backend = 'python'
        self.der.backend = backend
        self.avg.backend = backend
        self.dif.backend = backend
        self.dyn.backend = backend
        self.eul.backend = backend

        self.droplet (self.out_H)

        for i in range (2):
            #
            # run the stencil
            #
            self.der.run (in_V=self.out_U,
                          out_Vx=self.in_Ux,
                          out_Vy=self.in_Uy)
            self.der.run (in_V=self.out_V,
                          out_Vx=self.in_Vx,
                          out_Vy=self.in_Vy)
            self.der.run (in_V=self.out_H,
                          out_Vx=self.in_Hx,
                          out_Vy=self.in_Hy)

            for i in range (1, self.domain[0] - 1):
                for j in range (1, self.domain[1] - 1):
                    print ('%d %d\t%e' % (i, j, self.in_Ux[i,j,0]))

            self.avg.run (out_avg=self.avg_U,
                          in_V=self.out_U)
            self.avg.run (out_avg=self.avg_V,
                          in_V=self.out_V)
            self.avg.run (out_avg=self.avg_H,
                          in_V=self.out_H)

            self.dif.run (out_V=self.out_U,
                          in_avg=self.avg_U)
            self.dif.run (out_V=self.out_V,
                          in_avg=self.avg_V)
            self.dif.run (out_V=self.out_H,
                          in_avg=self.avg_H)

            self.dyn.run (out_Hd=self.out_Hd,
                          out_Ud=self.out_Ud,
                          out_Vd=self.out_Vd,
                          in_H=self.out_H,
                          in_U=self.out_U,
                          in_V=self.out_V,
                          in_Hx=self.in_Hx,
                          in_Ux=self.in_Ux,
                          in_Hy=self.in_Hy,
                          in_Vy=self.in_Vy)

            self.eul.run (out_H=self.out_H,
                          out_U=self.out_U,
                          out_V=self.out_V,
                          in_Hd=self.out_Hd,
                          in_Ud=self.out_Ud,
                          in_Vd=self.out_Vd)

        return 

        #
        # initialize plot
        #
        #plt.switch_backend ('agg')
        fig = plt.figure ( )
        ax  = fig.add_subplot (111,
                               projection='3d',
                               autoscale_on=False)
        X, Y = np.meshgrid (np.arange (self.domain[0]),
                            np.arange (self.domain[1]))
        def init_frame ( ):
            #im = ax.imshow (self.out_H[:,:,0])
            im = ax.plot_wireframe (X, Y, self.out_H[:,:,0],
                                    linewidth=1)
            return [im]

        #
        # animation update function
        #
        def draw_frame (framenumber):
            #
            # run the stencil
            #
            self.der.run (in_V=self.out_U,
                          out_Vx=self.in_Ux,
                          out_Vy=self.in_Uy)
            self.der.run (in_V=self.out_V,
                          out_Vx=self.in_Vx,
                          out_Vy=self.in_Vy)
            self.der.run (in_V=self.out_H,
                          out_Vx=self.in_Hx,
                          out_Vy=self.in_Hy)

            self.avg.run (out_avg=self.avg_U,
                          in_V=self.out_U)
            self.avg.run (out_avg=self.avg_V,
                          in_V=self.out_V)
            self.avg.run (out_avg=self.avg_H,
                          in_V=self.out_H)

            self.dif.run (out_V=self.out_U,
                          in_avg=self.avg_U)
            self.dif.run (out_V=self.out_V,
                          in_avg=self.avg_V)
            self.dif.run (out_V=self.out_H,
                          in_avg=self.avg_H)

            self.dyn.run (out_Hd=self.out_Hd,
                          out_Ud=self.out_Ud,
                          out_Vd=self.out_Vd,
                          in_H=self.out_H,
                          in_U=self.out_U,
                          in_V=self.out_V,
                          in_Hx=self.in_Hx,
                          in_Ux=self.in_Ux,
                          in_Hy=self.in_Hy,
                          in_Vy=self.in_Vy)

            self.eul.run (out_H=self.out_H,
                          out_U=self.out_U,
                          out_V=self.out_V,
                          in_Hd=self.out_Hd,
                          in_Ud=self.out_Ud,
                          in_Vd=self.out_Vd)
            #
            # reset if the system becomes unstable
            #
            if np.any (np.isnan (self.out_H)):
                self.setUp ( )
                print ("Reseting ...")

            #im = ax.imshow (self.out_H[:,:,0])
            ax.cla ( )
            im = ax.plot_wireframe (X, Y, self.out_H[:,:,0],
                                    linewidth=1)
            ax.set_xlim ( (0, self.domain[0] - 1) )
            ax.set_ylim ( (0, self.domain[1] - 1) )
            ax.set_zlim ( (0.9, 1.10) )

            return [im]

        anim = animation.FuncAnimation (fig,
                                        draw_frame,
                                        frames=range (2),
                                        interval=30,
                                        init_func=init_frame,
                                        blit=False)
        #anim.save ('shallow.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
        plt.show ( )


    def test_shallow_water_combination (self, backend='python'):
        self.swu = SW_U (self.domain)
        self.swv = SW_V (self.domain)
        self.swh = SW_H (self.domain)

        self.swu.set_halo ( (1, 1, 1, 1) )
        self.swv.set_halo ( (1, 1, 1, 1) )
        self.swh.set_halo ( (1, 1, 1, 1) )

        self.swu.backend = backend
        self.swv.backend = backend
        self.swh.backend = backend

        self.swu.run (out_U=self.out_U,
                      out_Ud=self.out_Ud,
                      out_Ux=self.in_Ux)
        self.swv.run (out_V=self.out_V,
                      out_Vd=self.out_Vd,
                      out_Vy=self.in_Vy)
        self.swh.run (out_H=self.out_H,
                      out_Hd=self.out_Hd,
                      out_U=self.out_U,
                      out_Ud=self.out_Ud,
                      out_V=self.out_V,
                      out_Vd=self.out_Vd,
                      in_Ux=self.in_Ux,
                      in_Vy=self.in_Vy)

    
    def test_shallow_water_combination_native (self):
        self.test_shallow_water_combination (backend='c++')


    @attr (lang='python')
    def test_python_execution (self):
        for i in range (2):
            self.stencil.run (out_H=self.out_H,
                              out_Hd=self.out_Hd,
                              out_U=self.out_U,
                              out_Ud=self.out_Ud,
                              out_V=self.out_V,
                              out_Vd=self.out_Vd,
                              in_Ux=self.in_Ux,
                              in_Vy=self.in_Vy)








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
        self.dt = 0.005
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
        H[rr<(self.domain[0]/10)**2] = 1.1 
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

        H[:,self.domain[0]-2] =  H[:,self.domain[0]-1]  
        U[:,self.domain[0]-2] =  U[:,self.domain[0]-1]  
        V[:,self.domain[0]-2] = -V[:,self.domain[0]-1]

        H[0,:] =  H[1,:]
        U[0,:] = -U[1,:]
        V[0,:] =  V[1,:]

        H[self.domain[0]-1,:] =  H[self.domain[0]-2,:]
        U[self.domain[0]-1,:] = -U[self.domain[0]-2,:]
        V[self.domain[0]-1,:] =  V[self.domain[0]-2,:]


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
            self.Ud[p] = self.Ud[p] - 1.2 * self.Hx[p]
            self.Vd[p] = self.Vd[p] - 1.2 * self.Hy[p]
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



class ShallowWater2DTest (CopyTest):
    """
    A test case for the shallow water stencil defined above.-
    """
    def setUp (self):
        logging.basicConfig (level=logging.DEBUG)

        self.domain = (16, 16, 1)

        self.params = ('out_H', 'out_U', 'out_V',
                       'in_H',  'in_U',  'in_V')

        self.temps  = ('Havg', 'Hd', 'Hx', 'Hy', 
                       'Uavg', 'Ud', 'Ux', 'Uy', 
                       'Vavg', 'Vd', 'Vx', 'Vy')

        self.out_H = np.ones  (self.domain)
        self.out_U = np.zeros (self.domain)
        self.out_V = np.zeros (self.domain)
        self.in_H  = np.ones  (self.domain)
        self.in_U  = np.zeros (self.domain)
        self.in_V  = np.zeros (self.domain)
        
        self.stencil = ShallowWater2D (self.domain)
        
        self.stencil.set_halo        ( (2, 2, 2, 2) )
        self.stencil.set_k_direction ("forward")


    @attr(lang='c++')
    def test_interactive_plot (self):
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
        # initialize 3D plot
        #
        fig = plt.figure ( )
        ax  = fig.add_subplot (111,
                               projection='3d',
                               autoscale_on=False)
        X, Y = np.meshgrid (np.arange (self.domain[0]),
                            np.arange (self.domain[1]))
        #
        # first animation frame
        #
        def init_frame ( ):
            im = ax.plot_wireframe (X, Y, self.out_H[:,:,0],
                                    linewidth=1)
            return [im]

        #
        # animation update function
        #
        def draw_frame (framenumber):
            if framenumber % 2 == 0:
                #
                # run the stencil
                #
                self.stencil.run (out_H=self.out_H,
                                  out_U=self.out_U,
                                  out_V=self.out_V,
                                  in_H=self.in_H,
                                  in_U=self.in_U,
                                  in_V=self.in_V)
            else:
                #
                # run the stencil with swapped pointers
                #
                self.stencil.run (out_H=self.in_H,
                                  out_U=self.in_U,
                                  out_V=self.in_V,
                                  in_H=self.out_H,
                                  in_U=self.out_U,
                                  in_V=self.out_V)
            #
            # reset if the system becomes unstable
            #
            if np.any (np.isnan (self.out_H)):
                self.setUp ( )
                print ("Reseting ...")

            ax.cla ( )
            im = ax.plot_wireframe (X, Y, self.out_H[:,:,0],
                                    linewidth=1)
            ax.set_xlim ( (0, self.domain[0] - 1) )
            ax.set_ylim ( (0, self.domain[1] - 1) )
            ax.set_zlim ( (0.9, 1.10) )

            return [im]

        anim = animation.FuncAnimation (fig,
                                        draw_frame,
                                        frames=range (1000),
                                        interval=30,
                                        init_func=init_frame,
                                        blit=False)
        #anim.save ('shallow.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
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

        for i in range (2):
            print (self.out_U[:,:,0])
            self.stencil.run (out_H=self.out_H,
                              out_U=self.out_U,
                              out_V=self.out_V)

        self.assertIsNotNone (self.out_H)
        self.assertIsNotNone (self.out_U)
        self.assertIsNotNone (self.out_V)
