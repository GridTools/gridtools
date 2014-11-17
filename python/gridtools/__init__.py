# -*- coding: utf-8 -*-
import sys
import ast
import warnings

import numpy as np

from gridtools.functor import StencilFunctor



class StencilInspector (ast.NodeVisitor):
    """
    Inspects the source code of a stencil definition using its AST.-
    """
    def __init__ (self, cls):
        """
        Creates an inspector object using the source code of a stencil:

            cls     a class extending the MultiStageStencil.-
        """
        from inspect import getsource

        if issubclass (cls, MultiStageStencil):
            super (StencilInspector, self).__init__ ( )
            self.name = "%sStencil" % cls.__name__.capitalize ( )
            self.src = getsource (cls)
            self.kernel_func = None
        else:
            raise TypeError ("Class must extend 'MultiStageStencil'")


    def analyze (self):
        """
        Analyzes the source code of this stencil.-
        """
        module = ast.parse (self.src)
        self.visit (module)
        if self.kernel_func is None:
            raise NameError ("Class must implement a 'kernel' function")


    def translate (self):
        """
        Translates this functor to C++, using the gridtools interface.-
        """
        from jinja2 import Environment, PackageLoader

        jinja_env = Environment (loader = PackageLoader ('gridtools',
                                                         'templates'))
        tpl = jinja_env.get_template ("functor.c")
        print (tpl.render (stencil=self,
                           functor=self.kernel_func))


    def visit_FunctionDef (self, node):
        """
        Looks for the stencil's entry function 'kernel' and validates it:

            node    a node from the AST.-
        """
        #
        # look for the 'kernel' function, which is the starting point 
        # of the stencil
        #
        if node.name == 'kernel':
            #
            # this function should not return anything
            #
            if node.returns is None:
                self.kernel_func = StencilFunctor (node)
                self.kernel_func.analyze_params ( )
                self.kernel_func.analyze_loops  ( )
                self.translate ( )
                #
                # continue traversing the AST
                #
                for n in node.body:
                    super (StencilInspector, self).visit (n)
            else:
                raise ValueError ("The 'kernel' function should return 'None'.")



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
        #
        # a default halo - it goes:
        #
        #   (halo in minus direction, 
        #    halo in plus direction,
        #    index of first interior element,
        #    index of last interior element,
        #    total length in dimension)
        #
        self.halo = (1, 1)


    def kernel (self):
        raise NotImplementedError ( )


    def set_output (self, np_arr):
        """
        Sets the received NumPy array as output for the stencil calculation:
     
            np_arr  NumPy array to use as the stencil's output.-
        """
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
        #
        # id() does not seem to work as expected
        #
        #if id (output_field) in self.out_arrs:
        return np.ndindex (*output_field.shape)

