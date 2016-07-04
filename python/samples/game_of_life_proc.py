import numpy as np
import matplotlib.pyplot as plt

from gridtools.stencil import Stencil


@Stencil.kernel
def game_of_life (out_X, in_X):
    """
    Game of life implemented as a procedural stencil
    """
    for p in Stencil.get_interior_points (out_X):

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

# Set stencil options
Stencil.set_halo( (1,1,1,1) )
Stencil.set_k_direction('forward')

# Run stencil for 100 generations
for t in range(100):
    game_of_life(out_X=out_X, in_X=in_X)
    in_X = out_X
