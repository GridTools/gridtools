import unittest
import logging

import numpy as np

from nose.plugins.attrib import attr

from gridtools.stencil import MultiStageStencil, StencilInspector




class Copy (MultiStageStencil):
    """
    Definition of a simple copy stencil, as in 'examples/copy_stencil.h'.-
    """
    def __init__ (self):
        super ( ).__init__ ( )

    def kernel (self, out_data, in_data):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (out_data):
              out_data[p] = in_data[p]



class CombinedStencilTest (unittest.TestCase):
    def _run (self):
        kwargs = dict ( )
        for p in self.params:
            kwargs[p] = getattr (self, p)
        self.stencil.run (**kwargs)


    def setUp (self):
        logging.basicConfig (level=logging.DEBUG)

        self.domain = (16, 16, 8)
        self.params = ('out_data', 'in_data')
        self.temps  = ( )

        self.out_data = np.zeros (self.domain)
        self.in_data  = np.random.rand (*self.domain)

        self.stencil = Copy ( )
        self.stencil.set_halo ( (1, 1, 1, 1) )
        self.stencil.set_k_direction ("forward")


    def test_identity_combination (self):
        combined = self.stencil.build ( )
        combined.backend = 'python'
        combined.run (out_data=self.out_data,
                      in_data=self.in_data)


    def test_double_combination (self):
        combined = self.stencil.build (in_data=self.stencil.build ( ))
        combined.backend = 'python'
        combined.run (out_data=self.out_data,
                      in_data=self.in_data)
