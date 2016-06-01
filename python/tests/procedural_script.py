# -*- coding: utf-8 -*-
import numpy as np
from gridtools.stencil import Stencil
#from Stencil import kernel



@Stencil.kernel
def copy_stencil (out_cpy, in_cpy):
    """
    This stencil comprises a single stage.-
    """
    #
    # iterate over the points, excluding halo ones
    #
    for p in Stencil.get_interior_points (out_cpy):
          out_cpy[p] = in_cpy[p]

domain = (4, 4, 4)

out_cpy = np.zeros (domain,
                    dtype=np.float64,
                    order='F')
out_cpy1 = np.array(out_cpy)
out_cpy2 = np.array(out_cpy)
#
# workaround because of a bug in the power (**) implemention of NumPy
#
in_cpy = np.random.random_integers (10, size=domain)
in_cpy = in_cpy.astype (np.float64)
in_cpy = np.asfortranarray (in_cpy)

in_cpy1 = np.array(in_cpy)
in_cpy2 = np.array(in_cpy)

#copy_stencil.set_halo ( (1, 1, 1, 1) )
#copy_stencil.set_k_direction ("forward")

print('Copy with halo:',Stencil.get_halo())
copy_stencil(out_cpy=out_cpy, in_cpy=in_cpy)
print(out_cpy == in_cpy)

Stencil.set_halo((1,1,1,1))
copy_stencil(out_cpy=out_cpy1, in_cpy=in_cpy1)
print('\nCopy with halo:',Stencil.get_halo())
print(out_cpy1 == in_cpy1)
