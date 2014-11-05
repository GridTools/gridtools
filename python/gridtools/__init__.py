# -*- coding: utf-8 -*-
import numpy as np



class MultiStageStencil ( ):
    """
    A base class for defining stencils involving several stages.
    All stencils should inherit for this class.-
    """
    def __init__ (self):
        #
        # the output NumPy arrays of this stencil
        #
        self.out_arrs = list ( )


    def at (self):
        #
        # should only be called with a NumPy array as 'self'
        #
        try:
            if self.flags:
                print ("NumPy array!")
        except AttributeError:
            print ("Call this method from a NumPy array")


    def set_output (self, np_arr):
        """
        Sets the received NumPy array as output for the stencil calculation:
     
            np_arr  NumPy array to use as the stencil's output.-
        """
        import types
        np_arr.at = types.MethodType (self.at, np_arr)

        self.out_arrs.append (id (np_arr.data))


    def get_interior_points (self, output_field, k_direction='forward'):
        """
        Returns an iterator over the 'output_field' without including the halo:

            output_field    a NumPy array which has been previously registered
                            as an output field with the 'set_output' function;
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward', 'backward' or
                            'parallel'.-
        """
        if id (output_field) in self.out_arrs:
            return np.nditer (output_field,
                              op_flags=['readwrite'])

