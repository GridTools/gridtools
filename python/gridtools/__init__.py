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
        import inspect

        if issubclass (cls, MultiStageStencil):
            super (StencilInspector, self).__init__ ( )
            self.src         = inspect.getsource (cls)
            self.name        = "%sStencil" % cls.__name__.capitalize ( )
            self.kernel_func = None
            self.hdr_file    = None
            self.cpp_file    = None
            self.make_file   = None
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
        Translates this functor to C++, using the gridtools interface.
        It returns a string pair of rendered (header, cpp, make) files.-
        """
        from jinja2 import Environment, PackageLoader

        jinja_env = Environment (loader = PackageLoader ('gridtools',
                                                         'templates'))
        header = jinja_env.get_template ("functor.h")
        cpp    = jinja_env.get_template ("stencil.cpp")
        make   = jinja_env.get_template ("Makefile")

        return (header.render (stencil=self,
                               functor=self.kernel_func),
                cpp.render (stencil=self),
                make.render (stencil=self))


    def compile (self):
        """
        Compiles the translated code to a shared library, ready to be used.-
        """
        from os       import write, close, path
        from tempfile import mkdtemp, mkstemp

        #
        # create temporary files for the generated code
        #
        tmp_dir = mkdtemp (prefix="__gridtools_")
        hdr_hdl, self.hdr_file = mkstemp (suffix=".h",
                                          prefix="%s_" % self.name,
                                          dir=tmp_dir)
        cpp_hdl, self.cpp_file = mkstemp (suffix=".cpp",
                                          prefix="%s_" % self.name,
                                          dir=tmp_dir)
        make_hdl, self.make_file = mkstemp (prefix="Makefile_",
                                            dir=tmp_dir)
        #
        # ... and populate them
        #
        print ("# Creating C++ code in [%s] ..." % tmp_dir)
        hdr_src, cpp_src, make_src = self.translate ( )
        write (hdr_hdl, hdr_src.encode ('utf-8'))
        write (cpp_hdl, cpp_src.encode ('utf-8'))
        write (make_hdl, make_src.encode ('utf-8'))
        close (hdr_hdl)
        close (cpp_hdl)
        close (make_hdl)

        #
        # before starting the compilation of the dynamic library
        #
        print ("# c++ -std=c++11 -DBACKEND_BLOCK -DFLOAT_PRECISION=8 -DCXX11_DISABLE -fopenmp -I%s -I${GRIDTOOLS_HOME}/include -I${GRIDTOOLS_HOME}/include/communication/high-level -isystem ${GRIDTOOLS_HOME}/fussion/include -o %s.o %s" % (tmp_dir,
                self.cpp_file,
                self.cpp_file))


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

