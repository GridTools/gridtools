# -*- coding: utf-8 -*-
import sys
import ast
import logging
import warnings
import redbaron

import numpy as np

from gridtools.symbol import StencilSymbols
from gridtools.functor import Functor




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
            # the domain dimensions over which this stencil operates
            #
            self.domain  = None
            #
            # the user's stencil code is kept here
            #
            try:
                self.src = inspect.getsource (cls)
            except TypeError:
                #
                # the code will not be available if it has been written
                # in an interactive interpreter
                #
                self.src = None
            #
            # symbols gathered from the AST of the user's stencil are kept here
            #
            self.symbols = StencilSymbols ( )
            self.stencil_scope = self.symbols.stencil_scope

            #
            # a list of functors of this stenctil;
            # the kernel function is the entry functor of any stencil
            #
            self.functors = list ( )
        else:
            raise TypeError ("Class %s must extend 'MultiStageStencil'" % cls)


    def analyze (self):
        """
        Analyzes the source code of this stencil as a first step 
        to generate its C++ counterpart.-
        """
        if self.src:
            module = ast.parse (self.src)
            self.visit (module)
            if len (self.functors) == 0:
                raise NameError ("Class must implement a 'kernel' function")
        else:
            #
            # if the source code is not available, we may infer the user is
            # running from an interactive session
            #
            raise RuntimeError ("Please save your stencil classes to a file before changing the backend.-")
        #
        # print out the discovered symbols if in DEBUG mode
        #
        if __debug__:
            logging.debug ("Symbols found after static code analysis:")
            self.stencil_scope.dump ( )


    def analyze_params (self, nodes):
        """
        Extracts the stencil parameters from the AST-node list 'nodes'.-
        """
        for n in nodes:
            #
            # do not add the 'self' parameter
            #
            if n.arg != 'self':
                #
                # parameters starting with the 'in_' prefix are considered 'read only'
                #
                read_only = n.arg.startswith ('in_')
                self.stencil_scope.add_parameter (n.arg,
                                                  read_only=read_only)


    def resolve (self, stencil_obj, **kwargs):
        """
        Attempts to aquire more information about the discovered symbols
        using runtime and parameter information:

            stencil_obj     the instance of the user's stencil;
            kwargs          the parameters with which is was started.-
        """
        if len (kwargs.keys ( )) > 0:
            self.domain = self.symbols.resolve_params (**kwargs)
        self.symbols.resolve (stencil_obj)


    def visit_Assign (self, node):
        """
        Extracts symbols appearing in assignments in the user's stencil code:

            node        a node from the AST.-
        """
        # 
        # expr = expr
        #
        if len (node.targets) > 1:
            raise RuntimeError ("Only one assignment per line is accepted.")
        else:
            lvalue = None
            lvalue_node = node.targets[0]
            #
            # attribute assignment
            #
            if isinstance (lvalue_node, ast.Attribute):
                lvalue = "%s.%s" % (lvalue_node.value.id,
                                    lvalue_node.attr)
            # 
            # parameter or local variable assignment
            # 
            elif isinstance (lvalue_node, ast.Name):
                lvalue = lvalue_node.id
            else:
                logging.debug ("Ignoring assignment at %d" % node.lineno)
                return
            #
            # a constant if its rvalue is a Num
            #
            if isinstance (node.value, ast.Num):
                rvalue = float (node.value.n)
                self.stencil_scope.add_constant (lvalue, rvalue)
                logging.debug ("Adding numeric constant '%s'" % lvalue)
            #
            # function calls are resolved later by name
            #
            elif isinstance (node.value, ast.Call):
                rvalue = None
                self.stencil_scope.add_constant (lvalue, rvalue)
                logging.debug ("Constant '%s' holds a function value" % lvalue)
            #
            # attributes are resolved later by name
            #
            elif isinstance (node.value, ast.Attribute):
                rvalue = '%s.%s' % (node.value.value.id,
                                    node.value.attr)
                self.stencil_scope.add_constant (lvalue, rvalue)
                logging.debug ("Constant '%s' holds an attribute value" % lvalue)
            else:
                #
                # we keep all other expressions and try to resolve them later
                #
                self.stencil_scope.add_constant (lvalue, None)
                logging.debug ("Constant '%s' will be resolved later" % lvalue)


    def visit_FunctionDef (self, node):
        """
        Looks for function definitions inside the user's stencil and classifies
        them accordingly:

            node    a node from the AST.-
        """
        #
        # the stencil constructor is the recommended place to define 
        # (pre-calculated) constants and temporary data fields
        #
        if node.name == '__init__':
            logging.debug ("Stencil constructor at %d" % node.lineno)
            #
            # should be a call to the parent-class constructor
            #
            for n in node.body:
                try:
                    parent_call = (isinstance (n.value, ast.Call) and 
                                   isinstance (n.value.func.value, ast.Call) and
                                   n.value.func.attr == '__init__')
                    if parent_call:
                        logging.debug ("Parent constructor call at %d" % n.value.lineno)
                        break
                except AttributeError:
                    parent_call = False
            #
            # inform the user if the call was not found
            #
            if not parent_call:
                raise ReferenceError ("Missing parent constructor call")
            #
            # continue traversing the AST of this function
            #
            for n in node.body:
                self.visit (n)
        #
        # the 'kernel' function is the starting point of the stencil
        #
        elif node.name == 'kernel':
            logging.debug ("Entry function 'kernel' found at %d" % node.lineno)
            #
            # this function should return 'None'
            #
            if node.returns is None:
                #
                # the parameters of the 'kernel' are the stencil
                # arguments in the generated code
                #
                self.analyze_params (node.args.args)
                #
                # continue traversing the AST
                #
                for n in node.body:
                    #
                    # looks for 'get_interior_points' comprehensions
                    # 
                    if isinstance (n, ast.For):
                        from random import choice
                        from string import digits

                        #
                        # the iteration should call 'get_interior_points'
                        #
                        call = n.iter
                        if (call.func.value.id == 'self' and 
                            call.func.attr == 'get_interior_points'):
                            #
                            # a random name for this functor
                            #
                            funct_name = 'functor_%s' % ''.join ([choice (digits) for n in range (4)])
                            
                            #
                            # create a new scope for the symbols of this functor
                            #
                            functor_scope = self.symbols.add_functor (funct_name)

                            #
                            # create the functor object
                            #
                            funct = Functor (funct_name,
                                             n,
                                             functor_scope,
                                             self.stencil_scope)
                            self.functors.append (funct)
                            logging.info ("Functor '%s' created" % funct.name)
            else:
                raise ValueError ("The 'kernel' function should return 'None'")



class MultiStageStencil ( ):
    """
    A base class for defining stencils involving several stages.
    All stencils should inherit for this class.-
    """
    def __init__ (self):
        #
        # a unique name for the stencil object
        #
        self.name = "%sStencil" % self.__class__.__name__.capitalize ( )

        #
        # defines the way to execute the stencil, one of 'python' or 'c++'
        #
        self.backend = "python"

        #
        # the inspector object is used to JIT-compile this stencil
        #
        self.inspector = StencilInspector (self.__class__)

        #
        # TODO a default halo - it goes:
        #
        #   (halo in minus direction, 
        #    halo in plus direction,
        #    index of first interior element,
        #    index of last interior element,
        #    total length in dimension)
        #
        self.halo = None

        #
        # these files are automatically generated compile time
        #
        self.lib_file  = None
        self.hdr_file  = None
        self.cpp_file  = None
        self.make_file = None

        #
        # a reference to the compiled dynamic library
        #
        self.lib_obj = None


    def compile (self):
        """
        Compiles the translated code to a shared library, ready to be used.-
        """
        from os         import write, path, getcwd, chdir
        from ctypes     import cdll
        from tempfile   import mkdtemp, mkstemp
        from subprocess import call

        #
        # create a temporary directory and files for the generated code
        #
        tmp_dir        = mkdtemp (prefix="__gridtools_")
        self.lib_file  = 'lib%s.so' % self.name.lower ( )
        self.hdr_file  = '%s.h' % self.name
        self.cpp_file  = '%s.cpp' % self.name
        self.make_file = 'Makefile'

        #
        # ... and populate it ...
        #
        logging.info ("Compiling C++ code in [%s]" % tmp_dir)
        hdr_src, cpp_src, make_src = self.translate ( )

        with open (path.join (tmp_dir, self.hdr_file), 'w') as hdr_hdl:
            hdr_hdl.write (hdr_src)

        with open (path.join (tmp_dir, self.cpp_file), 'w') as cpp_hdl:
            cpp_hdl.write (cpp_src)

        with open (path.join (tmp_dir, self.make_file), 'w') as make_hdl:
            make_hdl.write (make_src)

        #
        # ... before starting the compilation of the dynamic library
        #
        current_dir = getcwd ( )
        chdir (tmp_dir)
        call (["make", 
               "--silent", 
               "--file=%s" % self.make_file])
        chdir (current_dir)

        #
        # attach the library object
        #
        try:
            self.lib_obj = cdll.LoadLibrary ("%s" % path.join (tmp_dir, 
                                                               self.lib_file))
        except OSError:
            self.lib_obj = None
            raise RuntimeError ("Cannot load dynamically-compiled library")


    def get_interior_points (self, data_field, k_direction='forward', halo=(0,0,0,0)):
        """
        Returns an iterator over the 'data_field' without including the halo:

            data_field      a NumPy array;
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward', 'backward' or
                            'parallel';
            halo            a tupe defining a 2D halo for the given 'data_field' 
                            (+i, -i, +j, -j).-
        """
        try:
            if len (data_field.shape) != 3:
                raise ValueError ("Only 3D arrays are supported.")
            #
            # define the direction in 'k'
            #
            i_dim, j_dim, k_dim = data_field.shape
            if k_direction == 'forward' or k_direction == 'parallel':
                k_dim_start = 0
                k_dim_end   = k_dim
                k_dim_inc   = 1
            elif k_direction == 'backward':
                k_dim_start = k_dim - 1
                k_dim_end   = -1
                k_dim_inc   = -1
            else:
                logging.warning ("Unknown direction '%s'" % k_direction)

        except AttributeError:
            raise TypeError ("Calling 'get_interior_points' without a NumPy array")

        else:
            #
            # define the halo over i and j
            #
            if len (halo) == 4:
                start_i = 0 + halo[0]
                end_i   = i_dim + halo[1]
                start_j = 0 + halo[2]
                end_j   = j_dim + halo[3]
                #
                # return the coordinate tuples in the correct order
                #
                for i in range (start_i, end_i):
                    for j in range (start_j, end_j):
                        for k in range (k_dim_start, k_dim_end, k_dim_inc):
                            yield InteriorPoint ((i, j, k))
            else:
                raise ValueError ("Invalid halo: it should contain four values")


    def kernel (self, *args, **kwargs):
        """
        This function is the entry point of the stencil and 
        should be implemented by the user.-
        """
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

        #
        # run the selected backend version
        #
        logging.info ("Running in %s mode ..." % self.backend.capitalize ( ))
        if self.backend == 'c++':
            #
            # automatic compilation only if the library is not available
            #
            if self.lib_obj is None:
                #
                # try to resolve all symbols before compiling:
                # first with doing a static code analysis, ...
                #
                self.inspector.analyze ( )

                #
                # ... then including runtime information
                #
                self.inspector.resolve (self, **kwargs)
                
                #
                # generate the code of all functors in this stencil
                #
                try:
                    for func in self.inspector.functors:
                        func.generate_code (self.inspector.src)

                except Exception as e:
                    logging.error ("Error while generating code\n%s" % str (e))
                    raise e

                else:
                    #
                    # compile the generated code
                    #
                    try:
                        self.compile ( )
                    except RuntimeError:
                        logging.error ("Compilation failed")
                        return
            #
            # prepare the list of parameters to call the library function
            #
            lib_params = list (self.inspector.domain)

            #
            # extract the buffer pointers from the NumPy arrays
            #
            for p in self.inspector.stencil_scope.get_parameters ( ):
                if p.name in kwargs.keys ( ):
                    lib_params.append (kwargs[p.name].ctypes.data_as (ctypes.c_void_p))
                else:
                    logging.warning ("Parameter '%s' does not exist in the symbols table" % p.name)
            #
            # call the compiled stencil
            # 
            self.lib_obj.run (*lib_params)
        #
        # run in Python mode
        #
        elif self.backend == 'python':
            self.kernel (**kwargs)
        else:
            logging.warning ("Unknown backend [%s]" % self.backend)


    def translate (self):
        """
        Translates this functor to C++, using the gridtools interface.
        It returns a string pair of rendered (header, cpp, make) files.-
        """
        from os.path import basename
        from jinja2  import Environment, PackageLoader

        
        def join_with_prefix (a_list, prefix, attribute=None):
            """
            A custom filter for template rendering.-
            """
            if attribute is None:
                return ['%s%s' % (prefix, e) for e in a_list]
            else:
                return ['%s%s' % (prefix, getattr (e, attribute)) for e in a_list]
        #
        # initialize the template renderer environment
        #
        jinja_env = Environment (loader=PackageLoader ('gridtools',
                                                       'templates'))
        jinja_env.filters["join_with_prefix"] = join_with_prefix

        #
        # prepare the functor template
        #
        functor_tpl = jinja_env.get_template ("functor.h")

        #
        # render the source code for each of the functors
        #
        functor_src = ""
        for f in self.inspector.functors:
            params       = list (f.scope.get_parameters ( ))
            functor_src += functor_tpl.render (functor=f,
                                               params=params)
        #
        # instantiate each of the templates and render them
        #
        header = jinja_env.get_template ("stencil.h")
        cpp    = jinja_env.get_template ("stencil.cpp")
        make   = jinja_env.get_template ("Makefile")

        params = list (self.inspector.stencil_scope.get_parameters ( ))
        temps  = list (self.inspector.stencil_scope.get_temporaries ( ))

        return (header.render (namespace=self.name.lower ( ),
                               stencil=self.inspector,
                               params=params,
                               temps=temps,
                               params_temps=params + temps,
                               functor_src=functor_src,
                               functors=self.inspector.functors),
                cpp.render  (stencil=self,
                             params=params),
                make.render (stencil=self))
    


class InteriorPoint (tuple):
    """
    Represents the point within a NumPy array at the given coordinates.-
    """
    def __add__ (self, other):
        if len (self) != len (other):
            raise ValueError ("Points have different dimensions.")
        return tuple (map (sum, zip (self, other)))

    def __sub__ (self, other):
        raise NotImplementedError ("Offsets are not supported with '-'.")
