# -*- coding: utf-8 -*-
import sys
import ast
import logging
import warnings

import numpy as np

from gridtools.symbol import StencilSymbols
from gridtools.functor import Functor




class StencilInspector (ast.NodeVisitor):
    """
    Inspects the source code of a stencil definition using its AST.-
    """
    def __init__ (self, obj):
        """
        Creates an inspector object using the source code of a stencil:

            obj     an object extending the MultiStageStencil.-
        """
        import inspect

        if issubclass (obj.__class__, MultiStageStencil):
            super (StencilInspector, self).__init__ ( )

            #
            # a reference to the MultiStageStencil we have to inspect
            #
            self.stencil = obj
            #
            # the domain dimensions over which this stencil operates
            #
            self.domain  = None
            #
            # the user's stencil code is kept here
            #
            try:
                self.src = inspect.getsource (obj.__class__)
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
            raise TypeError ("Class %s must extend 'MultiStageStencil'" % obj.__class__)


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


    def resolve (self):
        """
        Attempts to aquire more information about the discovered symbols
        with runtime information of user's stencil instance.-
        """
        for s in self.stencil_scope.get_all ( ):
            #
            # unresolved symbols have 'None' as their value
            #
            if s.value is None:
                #
                # is this a stencil's attribute?
                #
                if 'self' in s.name:
                    attr  = s.name.split ('.')[1]
                    s.value = getattr (self.stencil, attr, None)

                    #
                    # NumPy arrays are considered temporary data fields
                    #
                    if isinstance (s.value, np.ndarray):
                        #
                        # update the symbol table in this scope
                        #
                        self.stencil_scope.add_temporary (s.name,
                                                          s.value)
                    else:
                        self.stencil_scope.add_constant (s.name, 
                                                         s.value)


    def resolve_params (self, **kwargs):
        """
        Attempts to aquire more information about the discovered parameters 
        using runtime information.
        """
        for k,v in kwargs.items ( ):
            if self.stencil_scope.is_parameter (k):
                if isinstance (v, np.ndarray):
                    #
                    # update the value of this parameter
                    #
                    self.stencil_scope.add_parameter (k,
                                                      v,
                                                      read_only=self.stencil_scope.is_parameter (k,
                                                                                                 read_only=True))
                    #
                    #
                    # check the dimensions of different parameters match
                    #
                    if self.domain is None:
                        self.domain = v.shape
                    elif self.domain != v.shape:
                        logging.warning ("Dimensions of parameter '%s':%s do not match %s" % (k, 
                                                                                              v.shape,
                                                                                              self.domain))
                else:
                    logging.warning ("Parameter '%s' is not a NumPy array" % k)


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

            rvalue_node = node.value
            #
            # a constant if its rvalue is a Num
            #
            if isinstance (rvalue_node, ast.Num):
                rvalue = float (rvalue_node.n)
                self.stencil_scope.add_constant (lvalue, rvalue)
                logging.debug ("Adding numeric constant '%s'" % lvalue)
            #
            # variable names are resolved using runtime information
            #
            elif isinstance (rvalue_node, ast.Name):
                try:
                    rvalue = eval (rvalue_node.id)
                    self.stencil_scope.add_constant (lvalue, rvalue)
                    logging.debug ("Adding constant '%s'" % lvalue)

                except NameError:
                    self.stencil_scope.add_constant (lvalue, None)
                    logging.debug ("Delayed resolution of constant '%s'" % lvalue)
            #
            # function calls are resolved later by name
            #
            elif isinstance (rvalue_node, ast.Call):
                rvalue = None
                self.stencil_scope.add_constant (lvalue, rvalue)
                logging.debug ("Constant '%s' holds a function value" % lvalue)
            #
            # attributes are resolved using runtime information
            #
            elif isinstance (rvalue_node, ast.Attribute):
                rvalue = getattr (eval (rvalue_node.value.id),
                                  rvalue_node.attr)
                self.stencil_scope.add_constant (lvalue, rvalue)
                logging.debug ("Constant '%s' holds an attribute value" % lvalue)
            #
            # try to discover the correct type using runtime information
            #
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
                            funct_name = 'functor_%03d' % len (self.symbols.functor_scopes)
                            
                            #
                            # create a new scope for the symbols of this functor
                            #
                            functor_scope = self.symbols.add_functor (funct_name)

                            #
                            # create a functor object
                            #
                            funct = Functor (funct_name,
                                             n,
                                             functor_scope,
                                             self.stencil_scope)
                            self.functors.append (funct)
                            logging.debug ("Functor '%s' created" % funct.name)

                            # 
                            # update halo information if present
                            #
                            for k in call.keywords:
                                if k.arg == 'halo':
                                    self.stencil.halo = tuple ([e.n for e in k.value.elts])
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
        self.inspector = StencilInspector (self)

        #
        # a halo descriptor - see 'set_halo' bellow
        #
        self.halo = (0, 0, 0, 0)

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
            halo            a tuple defining a 2D halo over the given 
                            'data_field'. See 'set_halo'.-
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
            self.set_halo (halo)

            start_i = 0 + self.halo[1]
            end_i   = i_dim - self.halo[0]
            start_j = 0 + self.halo[3]
            end_j   = j_dim - self.halo[2]
            #
            # return the coordinate tuples in the correct order
            #
            for i in range (start_i, end_i):
                for j in range (start_j, end_j):
                    for k in range (k_dim_start, k_dim_end, k_dim_inc):
                        yield InteriorPoint ((i, j, k))


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
                self.inspector.resolve ( )
                self.inspector.resolve_params (**kwargs)
                
                #
                # print out the discovered symbols if in DEBUG mode
                #
                if __debug__:
                    logging.debug ("Symbols found after using run-time resolution:")
                    self.inspector.stencil_scope.dump ( )
                    for f in self.inspector.functors:
                        f.scope.dump ( )

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


    def set_halo (self, halo=(0,0,0,0)):
        """
        Applies the received 'halo' setting, which is defined as

            (halo in negative direction over _i_, 
             halo in positive direction over _i_,
             halo in negative direction over _j_,
             halo in positive direction over _j_).-
        """
        if len (halo) == 4:
            if halo[0] >= 0 and halo[2] >= 0:
                if halo[1] >= 0 and halo[3] >= 0:
                    self.halo = halo
                else:
                    raise ValueError ("Invalid halo %s: definition for the positive halo should be zero or a positive integer")
            else:
                raise ValueError ("Invalid halo %s: definition for the negative halo should be zero or a positive integer")
        else:
            raise ValueError ("Invalid halo %s: it should contain four values")


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
                               stencil=self,
                               scope=self.inspector.stencil_scope,
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
