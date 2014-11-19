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
        The kernel function is the entry point of the stencil execution and
        should follow several conventions.-
        """
        #
        # FIXME will not work because the 'class' definition is indented and
        #       it should not be
        #
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

    def test_only_keyword_arguments (self):
        """
        When calling 'run' on a stencil, only keyword arguments should be used.-
        """
        domain = (128, 128, 60)
        output_field = np.zeros (domain)
        input_field = np.random.rand (*domain)
        copy = Copy ( )
        with self.assertRaises (KeyError):
            copy.run (output_field,
                      input_field)


    def test_python_execution (self):
        """
        Checks that the stencil results are correct if executing in Python mode.-
        """
        domain = (128, 128, 60)
        output_field = np.zeros (domain)
        input_field = np.random.rand (*domain)
        copy = Copy ( )
        copy.run (out_data=output_field,
                  in_data=input_field)
        self.assertTrue (np.array_equal (input_field, 
                                         output_field),
                         "Arrays should be equal")


    def test_native_execution (self):
        """
        Checks stencil compilation and execution from a dynamic library.
        Note that the Python code is practically identical, except for the
        call to the 'backend' attribute.
        It also checks that the stencil results are correct after execution.-
        """
        domain = (512, 512, 60)
        output_field = np.zeros (domain)
        input_field = np.random.rand (*domain)
        copy = Copy ( )
        copy.backend = 'c++'
        copy.run (out_data=output_field,
                  in_data=input_field)
        self.assertNotEqual (copy.inspector.lib_obj, None)
        self.assertTrue     ('_FuncPtr' in dir (copy.inspector.lib_obj))
        self.assertTrue (np.array_equal (input_field, 
                                         output_field),
                         "Arrays should be equal")

