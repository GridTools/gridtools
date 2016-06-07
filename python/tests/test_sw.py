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

from nose.plugins.attrib import attr

from gridtools.stencil   import Stencil, MultiStageStencil
from tests.test_stencils import CopyTest




class SW (MultiStageStencil):
    def __init__ (self, domain):
        super ( ).__init__ ( )
        #
        # constants to callibrate the system - working with (24, 24, 0) and +0.1 droplet
        #
        #self.bl     = 0.2
        #self.dt     = 0.001
        #self.growth = 0.5
        self.bl      = 0.2
        self.growth  = 1.2
        self.dt      = 0.05

        #
        # temporary data fields
        #
        self.Hd   = np.zeros (domain)
        self.Ud   = np.zeros (domain)
        self.Vd   = np.zeros (domain)
        self.Hx   = np.zeros (domain)
        self.Ux   = np.zeros (domain)
        self.Vx   = np.zeros (domain)
        self.Hy   = np.zeros (domain)
        self.Uy   = np.zeros (domain)
        self.Vy   = np.zeros (domain)

        self.L    = np.zeros (domain)
        self.R    = np.zeros (domain)
        self.T    = np.zeros (domain)
        self.B    = np.zeros (domain)

        self.Dh   = np.zeros (domain)
        self.Du   = np.zeros (domain)
        self.Dv   = np.zeros (domain)


    def stage_momentum (self, out_M, out_Md, out_Mx, out_My):
        for p in self.get_interior_points (out_M):
            self.L[p] = out_M[p + (-1,0,0)]
            self.R[p] = out_M[p + (1,0,0)]
            self.T[p] = out_M[p + (0,1,0)]
            self.B[p] = out_M[p + (0,-1,0)]

            out_Mx[p] = self.R[p] - self.L[p]
            out_My[p] = self.T[p] - self.B[p]

            out_Md[p] = out_M[p] * (1.0 - self.bl) + self.bl * (0.25 * (self.L[p] +
                                                                        self.R[p] +
                                                                        self.T[p] +
                                                                        self.B[p]))


    @Stencil.kernel
    def kernel (self, in_H, in_U, in_V, out_H, out_U, out_V):
        #
        # momentum calculation for each field
        #
        self.stage_momentum (out_M  = in_U,
                             out_Md = self.Ud,
                             out_Mx = self.Ux,
                             out_My = self.Uy)

        self.stage_momentum (out_M  = in_V,
                             out_Md = self.Vd,
                             out_Mx = self.Vx,
                             out_My = self.Vy)

        self.stage_momentum (out_M  = in_H,
                             out_Md = self.Hd,
                             out_Mx = self.Hx,
                             out_My = self.Hy)
        #
        # dynamics and momentum combined
        #
        for p in self.get_interior_points (out_H):
            self.Dh[p] = -self.Ud[p] * self.Hx[p] -self.Vd[p] * self.Hy[p] - self.Hd[p] * (self.Ux[p] + self.Vy[p])
            self.Du[p] = -self.Ud[p] * self.Ux[p] -self.Vd[p] * self.Uy[p] - self.growth * self.Hx[p]
            self.Dv[p] = -self.Ud[p] * self.Vx[p] -self.Vd[p] * self.Vy[p] - self.growth * self.Hy[p]
            #
            # take first-order Euler step
            #
            out_H[p] = self.Hd[p] + self.dt * self.Dh[p]
            out_U[p] = self.Ud[p] + self.dt * self.Du[p]
            out_V[p] = self.Vd[p] + self.dt * self.Dv[p]




class SWTest (CopyTest):
    def setUp (self):
        super ( ).setUp ( )

        self.domain = (64, 64, 1)

        self.params = ('in_H',
                       'in_U',
                       'in_V',
                       'out_H',
                       'out_U',
                       'out_V')
        self.temps  = ('self.Hd',
                       'self.Ud',
                       'self.Vd',
                       'self.Hx',
                       'self.Ux',
                       'self.Vx',
                       'self.Hy',
                       'self.Uy',
                       'self.Vy',
                       'self.Dh',
                       'self.Du',
                       'self.Dv')

        self.stencil = SW (self.domain)
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ('forward')

        self.out_H  = np.zeros (self.domain, order='F')
        self.out_H += 0.000001
        self.out_U  = np.zeros (self.domain, order='F')
        self.out_U += 0.000001
        self.out_V  = np.zeros (self.domain, order='F')
        self.out_V += 0.000001

        self.droplet (self.out_H)

        self.in_H   = np.copy (self.out_H)
        self.in_U   = np.copy (self.out_U)
        self.in_V   = np.copy (self.out_V)


    def droplet (self, H, val=1.0):
        """
        A two-dimensional falling drop into the water:

            H   the water height field.-
        """
        x,y = np.mgrid[:self.domain[0], :self.domain[1]]
        droplet_x, droplet_y = self.domain[0]/2, self.domain[1]/2
        rr = (x-droplet_x)**2 + (y-droplet_y)**2
        H[rr<(self.domain[0]/10.0)**2] = val


    @attr (lang='c++')
    def test_animation (self, nframe=200):
        try:
            import os
            import pyqtgraph.opengl as gl
            from   pyqtgraph.Qt import QtCore, QtGui
        except ImportError:
            self.skipTest ('Could not import required packages')
        else:
            #
            # make sure X11 is available
            #
            if os.environ.get ('DISPLAY') is None:
                self.skipTest ("no DISPLAY available")
            else:
                #
                # get a Qt application context
                #
                self.qt_app = QtGui.QApplication.instance ( )
                if self.qt_app is None:
                    self.qt_app = QtGui.QApplication ([])

                ## Create a GL View widget to display data
                w = gl.GLViewWidget()
                w.show()
                w.setWindowTitle ('GridTools example: Shallow Water equation')
                w.setCameraPosition(distance=90)

                ## Add a grid to the view
                g = gl.GLGridItem()
                g.setSize (x=self.domain[0] + 2,
                           y=self.domain[1] + 2)
                g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
                w.addItem(g)

                ## Animated example
                ## compute surface vertex data
                x = np.linspace (0, self.domain[0], self.domain[0]).reshape (self.domain[0], 1)
                y = np.linspace (0, self.domain[1], self.domain[1]).reshape (1, self.domain[1])

                ## create a surface plot, tell it to use the 'heightColor' shader
                ## since this does not require normal vectors to render (thus we
                ## can set computeNormals=False to save time when the mesh updates)
                self.p4 = gl.GLSurfacePlotItem (shader='heightColor',
                                                computeNormals=False,
                                                smooth=False)
                self.p4.shader()['colorMap'] = np.array([0.2, 1, 0.8, 0.2, 0.1, 0.1, 0.2, 0, 2])
                self.p4.translate (self.domain[0]/-2.0,
                                   self.domain[1]/-2.0,
                                   0)
                w.addItem(self.p4)

                self.frame = 0
                self.stencil.set_backend ('c++')

                def update ( ):
                    try:
                        if (self.stencil.dt * self.frame) % 5 == 0:
                            self.droplet (self.out_H, val=3.95)

                        self.in_H   = np.copy (self.out_H)
                        self.in_U   = np.copy (self.out_U)
                        self.in_V   = np.copy (self.out_V)

                        self.stencil.run (in_H=self.in_H,
                                          in_U=self.in_U,
                                          in_V=self.in_V,
                                          out_H=self.out_H,
                                          out_U=self.out_U,
                                          out_V=self.out_V)
                        self.frame += 1
                        self.p4.setData (z=self.out_H[:,:,0])

                    finally:
                        if self.frame < nframe:
                            QtCore.QTimer ( ).singleShot (10, update)
                        else:
                            self.qt_app.exit ( )
                update ( )
                self.qt_app.exec_ ( )


    def test_data_dependency_detection (self, expected_deps=None, backend='c++'):
        if expected_deps is None:
            expected_deps = [('self.L',  'in_H'),
                             ('self.R',  'in_H'),
                             ('self.T',  'in_H'),
                             ('self.B',  'in_H'),
                             ('self.L',  'in_U'),
                             ('self.R',  'in_U'),
                             ('self.T',  'in_U'),
                             ('self.B',  'in_U'),
                             ('self.L',  'in_V'),
                             ('self.R',  'in_V'),
                             ('self.T',  'in_V'),
                             ('self.B',  'in_V'),
                             ('self.Hx', 'self.R'),
                             ('self.Hx', 'self.L'),
                             ('self.Ux', 'self.R'),
                             ('self.Ux', 'self.L'),
                             ('self.Vx', 'self.R'),
                             ('self.Vx', 'self.L'),
                             ('self.Hy', 'self.T'),
                             ('self.Hy', 'self.B'),
                             ('self.Uy', 'self.T'),
                             ('self.Uy', 'self.B'),
                             ('self.Vy', 'self.T'),
                             ('self.Vy', 'self.B'),
                             ('self.Hd', 'in_H'),
                             ('self.Hd', 'self.L'),
                             ('self.Hd', 'self.R'),
                             ('self.Hd', 'self.T'),
                             ('self.Hd', 'self.B'),
                             ('self.Ud', 'in_U'),
                             ('self.Ud', 'self.L'),
                             ('self.Ud', 'self.R'),
                             ('self.Ud', 'self.T'),
                             ('self.Ud', 'self.B'),
                             ('self.Vd', 'in_V'),
                             ('self.Vd', 'self.L'),
                             ('self.Vd', 'self.R'),
                             ('self.Vd', 'self.T'),
                             ('self.Vd', 'self.B'),
                             ('self.Vd', 'self.B'),
                             ('self.Vd', 'self.B'),
                             ('out_H', 'self.Hd'),
                             ('out_H', 'self.Dh'),
                             ('out_U', 'self.Ud'),
                             ('out_U', 'self.Du'),
                             ('out_V', 'self.Vd'),
                             ('out_V', 'self.Dv')]
        super ( ).test_data_dependency_detection (deps=expected_deps,
                                                  backend=backend)


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS

        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_Hd',   None)
        self.add_expected_offset ('in_Ud',   None)
        self.add_expected_offset ('in_Vd',   None)
        self.add_expected_offset ('in_Hx',   None)
        self.add_expected_offset ('in_Ux',   None)
        self.add_expected_offset ('in_Vx',   None)
        self.add_expected_offset ('in_Hy',   None)
        self.add_expected_offset ('in_Uy',   None)
        self.add_expected_offset ('in_Vy',   None)
        self.add_expected_offset ('self.L',  None)
        self.add_expected_offset ('self.L',  None)
        self.add_expected_offset ('self.L',  None)
        self.add_expected_offset ('self.R',  None)
        self.add_expected_offset ('self.R',  None)
        self.add_expected_offset ('self.R',  None)
        self.add_expected_offset ('self.T',  None)
        self.add_expected_offset ('self.T',  None)
        self.add_expected_offset ('self.T',  None)
        self.add_expected_offset ('self.B',  None)
        self.add_expected_offset ('self.B',  None)
        self.add_expected_offset ('self.B',  None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Hx', None)
        self.add_expected_offset ('self.Hx', None)
        self.add_expected_offset ('self.Hx', None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Ux', None)
        self.add_expected_offset ('self.Ux', None)
        self.add_expected_offset ('self.Ux', None)
        self.add_expected_offset ('self.Vx', None)
        self.add_expected_offset ('self.Vx', None)
        self.add_expected_offset ('self.Hy', None)
        self.add_expected_offset ('self.Hy', None)
        self.add_expected_offset ('self.Uy', None)
        self.add_expected_offset ('self.Uy', None)
        self.add_expected_offset ('self.Vy', None)
        self.add_expected_offset ('self.Vy', None)
        self.add_expected_offset ('self.Vy', None)
        self.add_expected_offset ('self.Dh', None)
        self.add_expected_offset ('self.Du', None)
        self.add_expected_offset ('self.Dv', None)
        self.add_expected_offset ('out_H',   None)
        self.add_expected_offset ('out_U',   None)
        self.add_expected_offset ('out_V',   None)
        self.add_expected_offset ('in_H',   [-1,1,-1,1])
        self.add_expected_offset ('in_U',   [-1,1,-1,1])
        self.add_expected_offset ('in_V',   [-1,1,-1,1])

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)


    def test_ghost_cell_pattern (self):
        expected_patterns = [ [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0] ]
        super ( ).test_ghost_cell_pattern (expected_patterns,
                                           backend='c++')
        super ( ).test_ghost_cell_pattern (expected_patterns,
                                           backend='cuda')


    def test_interactive_plot (self):
        from shutil     import which
        from gridtools  import plt
        from matplotlib import animation

        #
        # need this program to create the animation
        #
        if which ('ffmpeg'):
            self.stencil.set_backend ('c++')

            plt.switch_backend ('agg')

            fig = plt.figure ( )
            ax  = fig.add_subplot (111,
                                   projection='3d',
                                   autoscale_on=False)
            X, Y = np.meshgrid (np.arange (self.out_H.shape[0]),
                                np.arange (self.out_H.shape[1]))

            def init_frame ( ):
                ax.grid      (False)
                ax.set_xlim  ( (0, self.domain[0] - 1) )
                ax.set_ylim  ( (0, self.domain[1] - 1) )
                ax.set_zlim  ( (0, 3.50) )
                im = ax.plot_wireframe (X, Y, self.out_H[:,:,0],
                                        linewidth=1)
                return [im]

            def draw_frame (frame):
                if (self.stencil.dt * frame) % 5 == 0:
                    self.droplet (self.out_H, val=3.95)

                self.stencil.run (out_H=self.out_H,
                                  out_U=self.out_U,
                                  out_V=self.out_V)
                ax.cla       ( )
                ax.grid      (False)
                ax.set_xlim  ( (0, self.domain[0] - 1) )
                ax.set_ylim  ( (0, self.domain[1] - 1) )
                ax.set_zlim  ( (0, 3.50) )
                im = ax.plot_wireframe (X, Y, self.out_H[:,:,0],
                                        linewidth=1)
                return [im]

            anim = animation.FuncAnimation (fig,
                                            draw_frame,
                                            frames=range (50),
                                            interval=10,
                                            init_func=init_frame,
                                            blit=False)
            anim.save ('/tmp/%s.mp4' % self.__class__,
                       fps=48,
                       extra_args=['-vcodec', 'libx264'])
        else:
            self.skipTest ("No ffmpeg detected")


    def test_minimum_halo_detection (self):
        super ( ).test_minimum_halo_detection ([1, 1, 1, 1])


    @attr(lang='python')
    def test_python_results (self):
        super ( ).test_python_results (out_param   = 'out_H',
                                       result_file = 'sw_001.npy')


    def test_get_interior_points_K_static (self):
        super ( ).test_get_interior_points_K_static (self.out_H)


    def test_get_interior_points_K_object (self):
        super ( ).test_get_interior_points_K_object (self.out_H)


    def test_get_interior_points_IJ_static (self):
        super ( ).test_get_interior_points_IJ_static (self.out_H)


    def test_get_interior_points_IJ_object (self):
        super ( ).test_get_interior_points_IJ_object (self.out_H)



class LocalSW (MultiStageStencil):
    def __init__ (self, domain):
        super ( ).__init__ ( )
        #
        # constants to callibrate the system - working with (24, 24, 0) and +0.1 droplet
        #
        #self.bl     = 0.2
        #self.dt     = 0.001
        #self.growth = 0.5
        self.bl      = 0.2
        self.growth  = 1.2
        self.dt      = 0.05

        #
        # temporary data fields
        #
        self.Hd   = np.zeros (domain)
        self.Ud   = np.zeros (domain)
        self.Vd   = np.zeros (domain)
        self.Hx   = np.zeros (domain)
        self.Ux   = np.zeros (domain)
        self.Vx   = np.zeros (domain)
        self.Hy   = np.zeros (domain)
        self.Uy   = np.zeros (domain)
        self.Vy   = np.zeros (domain)


    def stage_momentum (self, in_M, out_Md, out_Mx, out_My):
        for p in self.get_interior_points (in_M):
            L = in_M[p + (-1,0,0)]
            R = in_M[p + (1,0,0)]
            T = in_M[p + (0,1,0)]
            B = in_M[p + (0,-1,0)]

            out_Mx[p] = R - L
            out_My[p] = T - B

            out_Md[p] = in_M[p] * (1.0 - self.bl) + self.bl * (0.25 * (L
                                                                       + R
                                                                       + T
                                                                       + B))


    @Stencil.kernel
    def kernel (self, in_H, in_U, in_V, out_H, out_U, out_V):
        #
        # momentum calculation for each field
        #
        self.stage_momentum (in_M  = in_U,
                             out_Md = self.Ud,
                             out_Mx = self.Ux,
                             out_My = self.Uy)

        self.stage_momentum (in_M  = in_V,
                             out_Md = self.Vd,
                             out_Mx = self.Vx,
                             out_My = self.Vy)

        self.stage_momentum (in_M  = in_H,
                             out_Md = self.Hd,
                             out_Mx = self.Hx,
                             out_My = self.Hy)
        #
        # dynamics and momentum combined
        #
        for p in self.get_interior_points (out_H):
            Dh = -self.Ud[p] * self.Hx[p] - self.Vd[p] * self.Hy[p] - self.Hd[p] * (self.Ux[p] + self.Vy[p])
            Du = -self.Ud[p] * self.Ux[p] - self.Vd[p] * self.Uy[p] - self.growth * self.Hx[p]
            Dv = -self.Ud[p] * self.Vx[p] - self.Vd[p] * self.Vy[p] - self.growth * self.Hy[p]
            #
            # take first-order Euler step
            #
            out_H[p] = self.Hd[p] + self.dt * Dh
            out_U[p] = self.Ud[p] + self.dt * Du
            out_V[p] = self.Vd[p] + self.dt * Dv



class LocalSWTest (SWTest):
    def setUp (self):
        super ( ).setUp ( )

        self.temps  = ('self.Hd',
                       'self.Ud',
                       'self.Vd',
                       'self.Hx',
                       'self.Ux',
                       'self.Vx',
                       'self.Hy',
                       'self.Uy',
                       'self.Vy')

        self.stencil = LocalSW (self.domain)
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ('forward')


    def test_data_dependency_detection (self, expected_deps=None, backend='c++'):
        if expected_deps is None:
            expected_deps = [('self.Hx', 'in_H'),
                             ('self.Hy', 'in_H'),
                             ('self.Hd', 'in_H'),
                             ('self.Ux', 'in_U'),
                             ('self.Uy', 'in_U'),
                             ('self.Ud', 'in_U'),
                             ('self.Vx', 'in_V'),
                             ('self.Vy', 'in_V'),
                             ('self.Vd', 'in_V'),
                             ('out_H', 'self.Hd'),
                             ('out_H', 'self.Ud'),
                             ('out_H', 'self.Hx'),
                             ('out_H', 'self.Vd'),
                             ('out_H', 'self.Hy'),
                             ('out_H', 'self.Ux'),
                             ('out_H', 'self.Vy'),
                             ('out_U', 'self.Ud'),
                             ('out_U', 'self.Ud'),
                             ('out_U', 'self.Ux'),
                             ('out_U', 'self.Vd'),
                             ('out_U', 'self.Uy'),
                             ('out_U', 'self.Hx'),
                             ('out_V', 'self.Vd'),
                             ('out_V', 'self.Ud'),
                             ('out_V', 'self.Vx'),
                             ('out_V', 'self.Vy'),
                             ('out_V', 'self.Hy')]
        super ( ).test_data_dependency_detection (expected_deps=expected_deps,
                                                  backend=backend)


    def test_automatic_access_pattern_detection (self):
        from gridtools import BACKENDS

        #
        # fields and their ranges
        #
        self.add_expected_offset ('in_Hd',   None)
        self.add_expected_offset ('in_Ud',   None)
        self.add_expected_offset ('in_Vd',   None)
        self.add_expected_offset ('in_Hx',   None)
        self.add_expected_offset ('in_Ux',   None)
        self.add_expected_offset ('in_Vx',   None)
        self.add_expected_offset ('in_Hy',   None)
        self.add_expected_offset ('in_Uy',   None)
        self.add_expected_offset ('in_Vy',   None)
        self.add_expected_offset ('L',  None)
        self.add_expected_offset ('L',  None)
        self.add_expected_offset ('L',  None)
        self.add_expected_offset ('R',  None)
        self.add_expected_offset ('R',  None)
        self.add_expected_offset ('R',  None)
        self.add_expected_offset ('T',  None)
        self.add_expected_offset ('T',  None)
        self.add_expected_offset ('T',  None)
        self.add_expected_offset ('B',  None)
        self.add_expected_offset ('B',  None)
        self.add_expected_offset ('B',  None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Ud', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Vd', None)
        self.add_expected_offset ('self.Hx', None)
        self.add_expected_offset ('self.Hx', None)
        self.add_expected_offset ('self.Hx', None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Hd', None)
        self.add_expected_offset ('self.Ux', None)
        self.add_expected_offset ('self.Ux', None)
        self.add_expected_offset ('self.Ux', None)
        self.add_expected_offset ('self.Vx', None)
        self.add_expected_offset ('self.Vx', None)
        self.add_expected_offset ('self.Hy', None)
        self.add_expected_offset ('self.Hy', None)
        self.add_expected_offset ('self.Uy', None)
        self.add_expected_offset ('self.Uy', None)
        self.add_expected_offset ('self.Vy', None)
        self.add_expected_offset ('self.Vy', None)
        self.add_expected_offset ('self.Vy', None)
        self.add_expected_offset ('Dh', None)
        self.add_expected_offset ('Du', None)
        self.add_expected_offset ('Dv', None)
        self.add_expected_offset ('out_H',   None)
        self.add_expected_offset ('out_U',   None)
        self.add_expected_offset ('out_V',   None)
        self.add_expected_offset ('in_H',   [-1,1,-1,1])
        self.add_expected_offset ('in_U',   [-1,1,-1,1])
        self.add_expected_offset ('in_V',   [-1,1,-1,1])

        for backend in BACKENDS:
            self.stencil.set_backend (backend)
            self._run ( )
            self.automatic_access_pattern_detection (self.stencil)



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

        self.dx = 1.00      # discretization step in X
        self.dy = 1.00      # discretization step in Y
        self.dt = 0.02      # time discretization step
        self.g  = 9.81      # gravitational acceleration

        #
        # temporary data fields
        #
        self.Hx = np.zeros (self.domain)
        self.Ux = np.zeros (self.domain)
        self.Vx = np.zeros (self.domain)

        self.Hy = np.zeros (self.domain)
        self.Uy = np.zeros (self.domain)
        self.Vy = np.zeros (self.domain)


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


    def stage_first_x (self, out_H, out_U, out_V):
        """
        First half step (stage X direction)
        """
        for p in self.get_interior_points (out_U):
            # height
            self.Hx[p]  = ( out_H[p + (1,1,0)] + out_H[p + (0,1,0)] ) / 2.0
            self.Hx[p] -= ( out_U[p + (1,1,0)] - out_U[p + (0,1,0)] ) * ( self.dt / (2*self.dx) )

            # X momentum
            self.Ux[p]  = ( out_U[p + (1,1,0)] + out_U[p + (0,1,0)] ) / 2.0
            self.Ux[p] -=  ( ( (out_U[p + (1,1,0)]*out_U[p + (1,1,0)]) / out_H[p + (1,1,0)] +
                               (out_H[p + (1,1,0)]*out_H[p + (1,1,0)]) * (self.g / 2.0) ) -
                             ( (out_U[p + (0,1,0)]*out_U[p + (0,1,0)]) / out_H[p + (0,1,0)] +
                               (out_H[p + (0,1,0)]*out_H[p + (0,1,0)]) * (self.g / 2.0) )
                           ) * ( self.dt / (2*self.dx) )

            # Y momentum
            self.Vx[p]  = ( out_V[p + (1,1,0)] + out_V[p + (0,1,0)] ) / 2.0
            self.Vx[p] -= ( ( out_U[p + (1,1,0)] * out_V[p + (1,1,0)] / out_H[p + (1,1,0)] ) -
                            ( out_U[p + (0,1,0)] * out_V[p + (0,1,0)] / out_H[p + (0,1,0)] )
                          ) * ( self.dt / (2*self.dx) )


    def stage_first_y (self, out_H, out_U, out_V):
        """
        First half step (stage Y direction)
        """
        for p in self.get_interior_points (out_V):
            # height
            self.Hy[p]  = ( out_H[p + (1,1,0)] + out_H[p + (1,0,0)] ) / 2.0
            self.Hy[p] -= ( out_V[p + (1,1,0)] - out_V[p + (1,0,0)] ) * ( self.dt / (2*self.dy) )

            # X momentum
            self.Uy[p]  = ( out_U[p + (1,1,0)] + out_U[p + (1,0,0)] ) / 2.0
            self.Uy[p] -= ( ( out_V[p + (1,1,0)] * out_U[p + (1,1,0)] / out_H[p + (1,1,0)] ) -
                            ( out_V[p + (1,0,0)] * out_U[p + (1,0,0)] / out_H[p + (1,0,0)] )
                          ) * ( self.dt / (2*self.dy) )

            # Y momentum
            self.Vy[p]  = ( out_V[p + (1,1,0)] + out_V[p + (1,0,0)] ) / 2.0
            self.Vy[p] -= ( (out_V[p + (1,1,0)] * out_V[p + (1,1,0)]) / out_H[p + (1,1,0)] +
                            (out_H[p + (1,1,0)] * out_H[p + (1,1,0)]) * ( self.g / 2.0 ) -
                            (out_V[p + (1,0,0)] * out_V[p + (1,0,0)]) / out_H[p + (1,0,0)] +
                            (out_H[p + (1,0,0)] * out_H[p + (1,0,0)]) * ( self.g / 2.0 )
                          ) * ( self.dt / (2*self.dy) )


    @Stencil.kernel
    def kernel (self, out_H, out_U, out_V):
        self.stage_first_x (out_H=out_H,
                            out_U=out_U,
                            out_V=out_V)

        self.stage_first_y (out_H=out_H,
                            out_U=out_U,
                            out_V=out_V)
        #
        # second and final stage
        #
        for p in self.get_interior_points (out_H):
            # height
            out_H[p] -= ( self.Ux[p + (0,-1,0)] - self.Ux[p + (-1,-1,0)] ) * (self.dt / self.dx)
            out_H[p] -= ( self.Vy[p + (-1,0,0)] - self.Vy[p + (-1,-1,0)] ) * (self.dt / self.dx)

            # X momentum
            out_U[p] -= ( (self.Ux[p + (0,-1,0)]  * self.Ux[p + (0,-1,0)])  / self.Hx[p + (0,-1,0)] +
                          (self.Hx[p + (0,-1,0)]  * self.Hx[p + (0,-1,0)])  * ( self.g / 2.0 ) -
                          (self.Ux[p + (-1,-1,0)] * self.Ux[p + (-1,-1,0)]) / self.Hx[p + (-1,-1,0)] +
                          (self.Hx[p + (-1,-1,0)] * self.Hx[p + (-1,-1,0)]) * ( self.g / 2.0 )
                        ) * ( self.dt / self.dx )
            out_U[p] -= ( (self.Vy[p + (-1,0,0)]  * self.Uy[p + (-1,0,0)]  / self.Hy[p + (-1,0,0)]) -
                          (self.Vy[p + (-1,-1,0)] * self.Uy[p + (-1,-1,0)] / self.Hy[p + (-1,-1,0)])
                        ) * ( self.dt / self.dy )

            # Y momentum
            out_V[p] -= ( (self.Ux[p + (0,-1,0)]  * self.Vx[p + (0,-1,0)]  / self.Hx[p + (0,-1,0)]) -
                          (self.Ux[p + (-1,-1,0)] * self.Vx[p + (-1,-1,0)] / self.Hx[p + (-1,-1,0)])
                        ) * ( self.dt / self.dx )
            out_V[p] -= ( (self.Vy[p + (-1,0,0)]  * self.Vy[p + (-1,0,0)])  / self.Hy[p + (-1,0,0)] +
                          (self.Hy[p + (-1,0,0)]  * self.Hy[p + (-1,0,0)])  * ( self.g / 2.0 ) -
                          ( (self.Vy[p + (-1,-1,0)] * self.Vy[p + (-1,-1,0)]) / self.Hy[p + (-1,-1,0)] +
                            (self.Hy[p + (-1,-1,0)] * self.Hy[p + (-1,-1,0)]) * ( self.g / 2.0 ) )
                        ) * ( self.dt / self.dy )

