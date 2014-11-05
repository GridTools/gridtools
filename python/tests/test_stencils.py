import unittest
import numpy as np

from gridtools import MultiStageStencil



class Copy (MultiStageStencil):
    """
    Definition of a simple copy stencil, as in 'examples/copy_stencil.h'.-
    """
    def __init__ (self):
        super.__init__ (self)
        #
        # output fields should be declared in the constructor, 
        # with an 'out_' prefix
        #
        self.out_data = None

    def kernel (self, in_data):
        """
        This stencil comprises a single stage.-
        """
        #
        # iterate over the points, excluding halo ones
        #
        for p in self.get_interior_points (self.out_data,
                                           k_direction="forward"):
            self.out_data.at (p) = in_data.at (p)


class CopyStencilTest (unittest.TestCase):
    """
    A test case for the copy stencil defined above.-
    """
    def test_results (self):
        """
        Checks that the stencil results are correct.-
        """
        domain = (45, 30, 60)
        output_field = np.zeros (domain)
        input_field = np.random.rand (*domain)
        copy = Copy ( )
        copy.set_output (output_field)
        copy.kernel (input_field)
        self.assertEqual (input_field, output_field)

