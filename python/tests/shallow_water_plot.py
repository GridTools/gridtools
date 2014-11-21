import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D



class SWPlot(object):
    def __init__( self ):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )

        rng = np.arange( -5, 5, 0.25 )
        self.X, self.Y = np.meshgrid(rng,rng)

        self.heightR = np.zeros( self.X.shape )
        self.surf = self.ax.plot_surface( 
                self.X, self.Y, self.heightR, rstride=1, cstride=1, 
                cmap=cm.jet, linewidth=0, antialiased=False ) 
        self.fig.show()


def draw_frame (framenumber, swplot):
    t = framenumber / 30.
    data = np.sin(swplot.X - 0.05*t) * np.sin(swplot.Y - 0.25*t)
    swplot.ax.clear ( )
    swplot.surf = swplot.ax.plot_surface( 
            swplot.X, swplot.Y, data, rstride=1, cstride=1, 
            cmap=cm.jet, linewidth=0, antialiased=False ) 
    return swplot.surf,
       
plt.ion()
p = SWPlot()
X, Y = p.X, p.Y
Z = np.zeros_like(p.X)

anim = animation.FuncAnimation (p.fig,
                           draw_frame,
                           fargs=(p,),
                           interval=5,
                           blit=False)
                           
