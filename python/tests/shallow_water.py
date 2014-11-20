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
import numpy as np



# grid size
n = 64

# gravity
g = 9.8

# timestep
dt =0.02

# space step size (for u, v)
dx = 1.0
dy = 1.0

# number of drops
ndrops = 5

# drop interval
dropstep = 500

# data fields
H = np.ones((n+2, n+2))
U = np.zeros((n+2, n+2))
V = np.zeros((n+2, n+2))

Hx = np.zeros((n+1, n+1))
Ux = np.zeros((n+1, n+1))
Vx = np.zeros((n+1, n+1))

Hy = np.zeros((n+1, n+1))
Uy = np.zeros((n+1, n+1))
Vy = np.zeros((n+1, n+1))

# number of drops to break equilibrium
ndrop = np.ceil ( np.random.rand()*ndrops )

# number of time steps
nsteps = 1500

for nstep in range ( nsteps ):
    # random water drops
    if nstep % dropstep == 0:
        # TODO : implement droplet
        #w = size (D,1);
        #i = ceil (rand*(n-w))+(1:w);
        #j = ceil (rand*(n-w))+(1:w);
        #H (i,j) = H (i,j) + rand*D;
        pass

    # reflective boundary conditions
    H[:,0] = H [:,1];      U [:,0] = U [:,1];      V [:,0] = -V[:,1];
    H[:,n+1] = H [:,n+0];  U [:,n+1] = U [:,n+0];  V [:,n+1] = -V[:,n+0];
    H[0,:] = H [1,:];      U [0,:] = -U[1,:];      V [0,:] = V [1,:];
    H[n+1,:] = H [n+0,:];  U [n+1,:] = -U[n+0,:];  V [n+1,:] = V [n+0,:];

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

        # TODO update plot
        #if mod (nstep,nplotstep) == 0
        #    C = abs (U (i,j)) + abs (V (i,j));  % Color shows momemtum
        #    t = nstep*dt;
        #    tv = norm (C,'fro');
        #    set (surfplot,'zdata',H (i,j),'cdata',C);
        #    set (top,'string',sprintf ('t = %6 .2f,  tv = %6 .2f',t,tv))
        #    drawnow

        # reset if the system becomes unstable
        #if all (all (isnan (H))), break, end  % Unstable, restart

