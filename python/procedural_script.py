# -*- coding: utf-8 -*-
import numpy as np
from gridtools.stencil import stencil_kernel



@stencil_kernel
def copy_stencil (self, out_cpy, in_cpy):
    """
    This stencil comprises a single stage.-
    """
    #
    # iterate over the points, excluding halo ones
    #
    for p in self.get_interior_points (out_cpy):
          out_cpy[p] = in_cpy[p]

domain = (64, 64, 32)

out_cpy = np.zeros (domain,
                    dtype=np.float64,
                    order='F')
#
# workaround because of a bug in the power (**) implemention of NumPy
#
in_cpy = np.random.random_integers (10, size=domain)
in_cpy = in_cpy.astype (np.float64)
in_cpy = np.asfortranarray (in_cpy)

copy_stencil.set_halo ( (1, 1, 1, 1) )
copy_stencil.set_k_direction ("forward")

copy_stencil(out_cpy=out_cpy, in_cpy=in_cpy)

out_cpy = in_cpy
