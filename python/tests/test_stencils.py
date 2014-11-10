import unittest
import numpy as np

from gridtools import MultiStageStencil



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
        for p in self.get_interior_points (out_data,
                                           k_direction="forward"):
            out_data[p] = in_data[p]



class CopyStencilTest (unittest.TestCase):
    """
    A test case for the copy stencil defined above.-
    """
    def test_ast (self):
        """
        Checks the AST analysis of the source code of the stencil.-
        """
        from gridtools import StencilInspector

        # 
        # the inspector works on the class definition, not the object
        #
        i = StencilInspector (Copy)
        i.analyze ( )


    def test_results (self):
        """
        Checks that the stencil results are correct.-
        """
        domain = (45, 30, 60)
        output_field = np.zeros (domain)
        input_field = np.random.rand (*domain)
        copy = Copy ( )
        copy.set_output (output_field)
        copy.kernel (output_field,
                     input_field)
        self.assertTrue (np.array_equal (input_field, 
                                         output_field),
                         "Arrays should be equal")

