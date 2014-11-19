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
            #
            # a unique name for the stencil object
            #
            self.name = "%sStencil" % cls.__name__.capitalize ( )
            #
            # the domain dimensions over which this stencil operates
            #
            self.dimensions  = None
            #
            # the dynamically-generated source code is kept here
            #
            self.src = inspect.getsource (cls)
            #
            # the kernel functor is the stencil's entry point for execution
            #
            self.kernel_func = None
            #
            # automatically generated files at compile time
            #
            self.lib_file  = None
            self.hdr_file  = None
            self.cpp_file  = None
            self.make_file = None
            #
            # a reference to the compiled dynamic library
            #
            self.lib_obj = None
        else:
            raise TypeError ("Class %s must extend 'MultiStageStencil'" % cls)


    def analyze (self, **kwargs):
        """
        Analyzes the parameters and source code of this stencil.-
        """
        #
        # analyze the source code
        #
        module = ast.parse (self.src)
        self.visit (module)
        if self.kernel_func is None:
            raise NameError ("Class must implement a 'kernel' function")

        #
        # extract run-time information from the parameters, if available
        #
        if len (kwargs.keys ( )) > 0:
            #
            # dimensions of data fields (i.e., shape of the NumPy arrays)
            #
            for k,v in kwargs.items ( ):
                if k in self.kernel_func.params.keys ( ):
                    if isinstance (v, np.ndarray):
                        #
                        # check the dimensions of different fields match
                        #
                        self.kernel_func.params[k].dim = v.shape
                        if self.dimensions is None:
                            self.dimensions = v.shape
                        elif self.dimensions != v.shape:
                            warnings.warn ("dimensions of parameter [%s] do not match %s" % (k, self.dimensions),
                                           UserWarning)
                    else:
                        warnings.warn ("parameter [%s] is not a NumPy array" % k,
                                       UserWarning)
                else:
                    warnings.warn ("ignoring parameter [%s]" % k,
                                   UserWarning)


    def translate (self):
        """
        Translates this functor to C++, using the gridtools interface.
        It returns a string pair of rendered (header, cpp, make) files.-
        """
        from jinja2 import Environment, PackageLoader

        def join_with_prefix (a_list, prefix, attribute=None):
            """
            A custom filter for template rendering.-
            """
            if attribute is None:
                return ['%s%s' % (prefix, e) for e in a_list]
            else:
                return ['%s%s' % (prefix, getattr (e, attribute)) for e in a_list]

        jinja_env = Environment (loader=PackageLoader ('gridtools',
                                                       'templates'))
        jinja_env.filters["join_with_prefix"] = join_with_prefix

        header = jinja_env.get_template ("functor.h")
        cpp    = jinja_env.get_template ("stencil.cpp")
        make   = jinja_env.get_template ("Makefile")

        return (header.render (stencil=self,
                               functor=self.kernel_func),
                cpp.render  (stencil=self,
                             functor=self.kernel_func),
                make.render (stencil=self))


    def compile (self):
        """
        Compiles the translated code to a shared library, ready to be used.-
        """
        from os         import write, close, path, getcwd, chdir
        from ctypes     import cdll
        from tempfile   import mkdtemp, mkstemp
        from subprocess import call

        #
        # create temporary files for the generated code
        #
        tmp_dir = mkdtemp (prefix="__gridtools_")
        self.lib_file = "%s/lib%s.so" % (tmp_dir,
                                         self.name.lower ( ))
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
        print ("# Compiling C++ code in [%s] ..." % tmp_dir)
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
        current_dir = getcwd ( )
        chdir (tmp_dir)
        call (["make", "--silent", "--file=%s" % self.make_file])
        chdir (current_dir)
        #
        # attach the library object
        #
        self.lib_obj = cdll.LoadLibrary ("%s" % self.lib_file)


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
        # defines the way to execute the stencil, one of 'python' or 'c++'
        #
        self.backend = "python"
        #
        # the inspector object is used to JIT-compile this stencil
        #
        self.inspector = StencilInspector (self.__class__)

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


    def kernel (self, *args, **kwargs):
        raise NotImplementedError ( )


    def run (self, *args, **kwargs):
        """
        Starts the execution of the stencil.-
        """
        import ctypes

        #
        # we only accept keyword arguments to avoid confusion
        #
        if len (args) > 0:
            raise KeyError ("Only keyword arguments are accepted.-")
        else:
            self.inspector.analyze (**kwargs)
        #
        # run the selected backend version
        #
        print ("# Running in %s mode ..." % self.backend.capitalize ( ))
        if self.backend == 'python':
            self.kernel (**kwargs)
        elif self.backend == 'c++':
            self.inspector.compile ( )
            #
            # extract the buffer pointers from the parameters (NumPy arrays)
            #
            params = list (self.inspector.dimensions)
            functor_params = self.inspector.kernel_func.params
            for k in sorted (functor_params,
                             key=lambda name: functor_params[name].id):
                if k in kwargs.keys ( ):
                    params.append (kwargs[k].ctypes.data_as (ctypes.c_void_p))
                else:
                    warnings.warn ("missing parameter [%s]" % k,
                                   UserWarning)
            #
            # call the compiled stencil
            #
            self.inspector.lib_obj.run (*params)
        else:
            warnings.warn ("unknown backend [%s]" % self.backend,
                           UserWarning)


    def get_interior_points (self, output_field, k_direction='forward'):
        """
        Returns an iterator over the 'output_field' without including the halo:

            output_field    a NumPy array which has been previously registered
                            as an output field with the 'set_output' function;
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward', 'backward' or
                            'parallel'.-
        """
        return np.ndindex (*output_field.shape)

