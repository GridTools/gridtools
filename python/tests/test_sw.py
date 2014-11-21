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
import numpy as np

from gridtools import MultiStageStencil, StencilInspector



class ShallowWater (MultiStageStencil):
    """
    Implements the shallow water equation as a multi-stage stencil.-
    """
    def __init__ (self):
        super (ShallowWater, self).__init__ ( )
        #
        # grid size
        #
        self.n = 64

        #
        # gravity-accelleration constant
        #
        self.g = 9.8

        # timestep
        self.dt =0.02

        # space step size (for u, v)
        self.dx = 1.0
        self.dy = 1.0

        # one drop to disturb the water
        self.drop = self.droplet (2, 11)


    def droplet (self, height, width):
        """
        A two-dimensional Gaussian of the falling drop into the water.-
        """
        x = np.array ([np.arange (-1, 1 + 2/(width-1), 2/(width-1))] * (width-1))
        y = np.copy (x)
        return height * np.exp (-5*(x**2 + y**2));


    def kernel (self, out_H, in_H):
        """
        This stencil comprises multiple stage.-
        """
        #
        # temporary data fields
        #
        tmp_U = np.zeros_like (out_H)
        tmp_V = np.zeros_like (out_H)

        tmp_Hx = np.zeros (np.subtract (out_H.shape), 1)
        tmp_Ux = np.zeros (np.subtract (out_H.shape), 1)
        tmp_Vx = np.zeros (np.subtract (out_H.shape), 1)

        tmp_Hy = np.zeros (np.subtract (out_H.shape), 1)
        tmp_Uy = np.zeros (np.subtract (out_H.shape), 1)
        tmp_Vy = np.zeros (np.subtract (out_H.shape), 1)

        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_data,
                                           k_direction="forward"):
            out_data[p] = in_data[p]







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





#
# the shallow water stencil object
#
sw = ShallowWater ( )

#
# initialize 3D plot
#
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure ( )
ax  = fig.add_subplot (111, 
                       projection='3d')

rng = np.arange (sw.n + 2)
X, Y = np.meshgrid (rng, rng)
surf = ax.plot_surface (X, Y, sw.H,
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
    swobj.step (framenumber)
    ax.clear ( )
    surf = ax.plot_surface (X, Y, swobj.H, 
                            rstride=1,
                            cstride=1, 
                            cmap=cm.jet, 
                            linewidth=0, 
                            antialiased=False)
    plt.savefig ("/tmp/water_%04d" % framenumber)
    return surf,

plt.ion ( )
anim = animation.FuncAnimation (fig,
                                draw_frame,
                                fargs=(sw,),
                                interval=2,
                                blit=False)
                           
