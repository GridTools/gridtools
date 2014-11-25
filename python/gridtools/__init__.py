# -*- coding: utf-8 -*-
import sys
import ast
import logging
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
            # the user's stencil code is kept here
            #
            self.src = inspect.getsource (cls)
            #
            # symbols gathered from the AST of the user's stencil
            #
            self.symbols = dict ( )
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


    def visit_Assign (self, node):
        """
        Looks for symbols, i.e. constants and data fields, in the user's
        stencil code:

            node        a node from the AST;
            ctx_func    the function within which the assignments are found.-
        """
        # 
        # only expr = expr
        # 
        if len (node.targets) > 1:
            raise RuntimeError ("Only one assignment per line is accepted.")
        else:
            lvalue = None
            rvalue = None
            lvalue_node = node.targets[0]

            if isinstance (lvalue_node, ast.Attribute):
                lvalue = lvalue_node.attr
            elif isinstance (lvalue_node, ast.Name):
                lvalue = lvalue_node.id
            else:
                logging.warning ("Ignoring assignment at %d" % node.lineno)
                return
            #
            # we consider it a constant iif its rvalue is a Num
            #
            if isinstance (node.value, ast.Num):
                rvalue = float (node.value.n)
                logging.info ("Found constant '%s' with value %.3f at %d" % (lvalue, 
                                                                             rvalue,
                                                                             node.lineno))
            else:
                #
                # otherwise, we keep it for resolution at run time
                #
                logging.warning ("Delaying resolution of '%s' at %d" % (lvalue,
                                                                        node.lineno))
            #
            # keep it in the dictionary for later use
            #
            assert lvalue != None, "Assignment's lvalue is None"
            self.symbols[lvalue] = rvalue


    def visit_FunctionDef (self, node):
        """
        Looks for function definitions inside the user's stencil and classifies
        them accordingly:

            node    a node from the AST.-
        """
        #
        # the stencil's constructor is the recommended place to define 
        # (run-time) constants and temporary fields
        #
        if node.name == '__init__':
            logging.info ("Found stencil constructor at %d" % node.lineno)
            #
            # should be a call to the parent's constructor
            #
            for n in node.body:
                try:
                    parent_call = (isinstance (n.value, ast.Call) and 
                                   isinstance (n.value.func.value, ast.Call) and
                                   n.value.func.attr == '__init__')
                    if parent_call:
                        logging.info ("Found parent's constructor call at %d" % 
                                      n.value.lineno)
                        break

                except AttributeError:
                    parent_call = False
            #
            # inform the user if the call was not found
            #
            if not parent_call:
                raise ReferenceError ("Missing parent's constructor call")
            #
            # continue traversing the AST of this function
            #
            for n in node.body:
                self.visit (n)
        #
        # the 'kernel' function is the starting point of the stencil
        #
        elif node.name == 'kernel':
            logging.info ("Found 'kernel' function at %d" % node.lineno)
            #
            # this function should return 'None'
            #
            if node.returns is None:
                self.kernel_func = StencilFunctor (node,
                                                   self.symbols)
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
            logging.info ("Symbols found: %s" % self.inspector.symbols)
        #
        # run the selected backend version
        #
        logging.info ("Running in %s mode ..." % self.backend.capitalize ( ))
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


    def get_interior_points (self, data_field, k_direction='forward'):
        """
        Returns an iterator over the 'data_field' without including the halo:

            data_field      a NumPy array;
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward', 'backward' or
                            'parallel'.-
        """
        try:
            if len (data_field.shape) != 3:
                raise ValueError ("Only 3D arrays are supported.")
            #
            # define the direction in 'k'
            #
            i_dim, j_dim, k_dim = data_field.shape
            if k_direction == 'forward':
                k_dim_start = 0
                k_dim_end   = k_dim
                k_dim_inc   = 1
            elif k_direction == 'backward':
                k_dim_start = k_dim - 1
                k_dim_end   = -1
                k_dim_inc   = -1
            else:
                warnings.warn ("unknown direction '%s'" % k_direction,
                              UserWarning)
            #
            # return the coordinate tuples in the correct order
            #
            for i in range (i_dim):
                for j in range (j_dim):
                    for k in range (k_dim_start, k_dim_end, k_dim_inc):
                            yield InteriorPoint ((i, j, k))

        except AttributeError:
            warings.warn ("calling 'get_interior_points' without a NumPy array",
                          UserWarning)



class InteriorPoint (tuple):
    """
    Represents the point within a NumPy array at the given coordinates.-
    """
    def __add__ (self, other):
        if len (self) != len (other):
            raise ValueError ("Points have different dimensions.")
        return tuple (map (sum, zip (self, other)))

