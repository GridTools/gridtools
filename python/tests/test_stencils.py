import unittest
import numpy as np

from gridtools import MultiStageStencil, StencilInspector



class Copy (MultiStageStencil):
    """
    Definition of a simple copy stencil, as in 'examples/copy_stencil.h'.-
    """
    def __init__ (self):
        super (Copy, self).__init__ ( )

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
    def test_extends (self):
        """
        A user-defined stencil should inherit from the MultiStageStencil class.-
        """
        with self.assertRaises (TypeError):
            class DoesNotExtendAndShouldFail (object):
                pass
            insp = StencilInspector (DoesNotExtendAndShouldFail)

        insp = StencilInspector (Copy)
        insp.analyze ( )
        self.assertNotEqual (insp, None)


    def test_kernel_function (self):
        """
        The kernel function si the entry point of the stencil execution and
        should follow several conventions.-
        """

        """
        with self.assertRaises (NameError):
            class KernelFunctionMissing (MultiStageStencil):
                def some_func (self):
                    return None

            insp = StencilInspector (KernelFunctionMissing)
            insp.analyze ( )

        with self.assertRaises (ValueError):
            class KernelFunctionShouldReturnNone (MultiStageStencil):
                def kernel (self):
                    return "something"
            insp = StencilInspector (KernelFunctionDoesNotReturnNone)
            insp.analyze ( )
        """
        pass


    def test_compile (self):
        """
        Compiles the generated code to a dynamic library.-
        """
        i = StencilInspector (Copy)
        i.analyze ( )
        i.compile ( )


    def test_ast (self):
        """
        Checks the AST analysis of the source code of the stencil.-
        """
        # 
        # the inspector works on the class definition, not the object
        #
        i = StencilInspector (Copy)
        i.analyze ( )
        print (i.translate ( ))


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

