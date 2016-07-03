import numpy as np
import matplotlib.pyplot as plt

from gridtools.stencil import Stencil, MultiStageStencil


class GameOfLife (MultiStageStencil):
    """
    Game of life implemented as a stencil
    """
    def __init__ (self):
        super ( ).__init__ ( )


    @Stencil.kernel
    def kernel (self, out_X, in_X):
        for p in self.get_interior_points (out_X):

            Y = in_X[p + (1,0,0)]  + in_X[p + (1,1,0)]   + \
                in_X[p + (0,1,0)]  + in_X[p + (-1,1,0)]  + \
                in_X[p + (-1,0,0)] + in_X[p + (-1,-1,0)] + \
                in_X[p + (0,-1,0)] + in_X[p + (1,-1,0)]

            out_X[p] = (in_X[p] and (Y == 2)) or (Y == 3)


# Generate a random initial population
domain = (50, 50, 4)
in_X = np.zeros(domain)
in_X[19:33,19:33,:] = np.random.rand(14,14,4) > .75
out_X = np.copy(in_X)

# Initialize stencil
stencil = GameOfLife()
stencil.set_halo( (1,1,1,1) )
stencil.set_k_direction('forward')

# Run stencil for 100 generations
for t in range(100):
    stencil.run(out_X=out_X, in_X=in_X)
    in_X = out_X
