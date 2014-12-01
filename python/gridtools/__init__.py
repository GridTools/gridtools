# -*- coding: utf-8 -*-
import sys
import ast
import logging
import warnings

import numpy as np

from gridtools.functor import StencilFunctor



class StencilSymbols (object):
    """
    Keeps a repository of all symbols, e.g., constants, aliases, temporary
    data fields, functor parameters, appearing in a user-defined stencil.-
    """
    def __init__ (self):
        #
        # we categorize all symbols in these groups
        #
        self.groups = set (('constant',    # anything replaceable at runtime;
                            'alias',       # variable aliasing;
                            'temp_field',  # temporary data fields;
                           ))              # functor parameters are identified
                                           # by the functor's name 
        #
        # initialize the container (k=group, v=dict), where
        # dict is k=symbol name, v=symbol value
        #
        self.symbols = dict ( )
        for g in self.groups:
            self.symbols[g] = dict ( )


    def __delitem__ (self, name):
        """
        Removes the symbol entry 'name' from this instance.-
        """
        for g in self.groups:
            if name in self.symbols[g].keys ( ):
                return self.symbols[g].__delitem__ (name)
        raise KeyError ("No symbol named '%s'")


    def __getitem__ (self, name):
        """
        Returns the value of symbol 'name' or None if not found.-
        """
        for g in self.groups:
            if name in self.symbols[g].keys ( ):
                return self.symbols[g][name]
        return None
     

    def _add (self, name, value, group):
        """
        Adds the received symbol into the corresponding container:

            name    name of this symbol;
            value   value of this symbol;
            group   container group into which to add the symbol.-
        """
        if group in self.groups:
            if name in self.symbols[group].keys ( ):
                logging.info ("Updated value of symbol '%s'" % name)
            self.symbols[group][name] = value


    def add_alias (self, name, value):
        """
        Adds an alias to the stencil's symbols:

            name    name of this symbol;
            value   value of this symbol.-
        """
        logging.info ("Alias '%s' points to '%s'" % (name,
                                                     str (value)))
        self._add (name, str (value), 'alias')


    def add_constant (self, name, value):
        """
        Adds a constant to the stencil's symbols:

            name    name of this symbol;
            value   value of this symbol.-
        """
        if value is None:
            logging.info ("Constant '%s' will be resolved later" % name)
        else:
            try:
                value = float (value)
                logging.info ("Constant '%s' has value %.3f" % (name,
                                                                value))
            except TypeError:
                if isinstance (value, np.ndarray):
                    logging.info ("Constant '%s' is a NumPy array %s" % (name,
                                                                         value.shape))
                else:
                    logging.info ("Constant '%s' has value %s" % (name,
                                                                  value))
        #
        # add the constant as a stencil symbol
        #
        self._add (name, value, 'constant')


    def add_functor (self, name):
        """
        Returns a new dictionary for keeping the functor's parameters:

            name    a unique name identifying the functor.-
        """
        if name in self.groups:
            raise NameError ("Functor '%s' already exists in symbol table.-")
        else:
            self.groups.add (name)
            self.symbols[name] = dict ( )
            return self.symbols[name]


    def add_temporary (self, name, value):
        """
        Adds a temporary data field stencil's symbols:

            name    name of this symbol;
            value   value of this symbol (a NumPy array).-
        """
        if value is None:
            raise ValueError ("Value of temporary field '%s' is None" % name)
        elif isinstance (value, np.ndarray):
            #
            # add the field as a temporary
            #
            self._add (name, value, 'temp_field')
            logging.info ("Temporary '%s' is a NumPy array %s" % (name,
                                                                  value.shape))
        else:
            raise TypeError ("Value of temporary field '%s' should be a NumPy array" % name)


    def items (self):
        """
        Returns all symbols in as (key, value) pairs.-
        """
        for g in self.groups:
            keys = self.symbols[g].keys ( )
            vals = self.symbols[g].values ( )
            for k,v in zip (keys, vals):
                yield (k, v)



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
            self.symbols = StencilSymbols ( )
            #
            # a list of functors of this stenctil;
            # the kernel function is the entry functor of any stencil
            #
            self.functors = list ( )
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
        if len (self.functors) == 0:
            raise NameError ("Class must implement a 'kernel' function")
        #
        # print out the discovered symbols
        #
        logging.debug ("Symbols found after static code analysis:")
        for k,v in self.symbols.items ( ):
            logging.debug ("\t%s:\t%s" % (k, str (v)))

        #
        # extract run-time information from the parameters, if available
        #
        if len (kwargs.keys ( )) > 0:
            #
            # dimensions of data fields (i.e., shape of the NumPy arrays)
            #
            for k,v in kwargs.items ( ):
                if k in self.functors[0].params.keys ( ):
                    if isinstance (v, np.ndarray):
                        #
                        # check the dimensions of different fields match
                        #
                        self.functors[0].params[k].dim = v.shape
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
                               functor=self.functors[0]),
                cpp.render  (stencil=self,
                             functor=self.functors[0]),
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
        try:
            self.lib_obj = cdll.LoadLibrary ("%s" % self.lib_file)
        except OSError:
            logging.error ("Cannot load library")
            raise RuntimeError


    def visit_Assign (self, node):
        """
        Looks for symbols in the user's stencil code:

            node        a node from the AST;
            ctx_func    the function within which the assignments are found.-
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
            # attribute or variable assignments only
            # 
            if isinstance (lvalue_node, ast.Attribute):
                lvalue = "%s.%s" % (lvalue_node.value.id,
                                    lvalue_node.attr)
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
                self.symbols.add_constant (lvalue, rvalue)
            #
            # function calls are resolved later by name
            #
            elif isinstance (node.value, ast.Call):
                rvalue = None
                self.symbols.add_constant (lvalue, rvalue)
            #
            # attribute aliases are resolved later by name
            #
            elif isinstance (node.value, ast.Attribute):
                rvalue = '%s.%s' % (node.value.value.id,
                                    node.value.attr)
                self.symbols.add_alias (lvalue, rvalue)
            else:
                logging.warning ("Don't know what to do with '%s' at %d" % (node.value,
                                                                            node.lineno))


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
            logging.info ("Stencil constructor at %d" % node.lineno)
            #
            # should be a call to the parent's constructor
            #
            for n in node.body:
                try:
                    parent_call = (isinstance (n.value, ast.Call) and 
                                   isinstance (n.value.func.value, ast.Call) and
                                   n.value.func.attr == '__init__')
                    if parent_call:
                        logging.info ("Parent's constructor call at %d" % n.value.lineno)
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
            logging.info ("Entry 'kernel' function at %d" % node.lineno)
            #
            # this function should return 'None'
            #
            if node.returns is None:
                #
                # a unique name for this functor
                #
                name = "%s_functor" % node.name
                #
                # the functor parameters will be kept here
                #
                funct_params = self.symbols.add_functor (name)
                #
                # create the functor object and start analyzing it
                #
                funct = StencilFunctor ("%s_functor" % node.name,
                                        node,
                                        funct_params,
                                        self.symbols)
                funct.analyze_params ( )
                funct.analyze_loops  ( )
                self.functors.append (funct)
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
        # TODO a default halo - it goes:
        #
        #   (halo in minus direction, 
        #    halo in plus direction,
        #    index of first interior element,
        #    index of last interior element,
        #    total length in dimension)
        #
        self.halo = None


    def _resolve (self, symbols):
        """
        Attempts to resolve missing symbols values with run-time information:

            symbols     the symbols dictionary.-
        """
        #
        # we cannot change the symbol table while looping over it,
        # so we save the changes here and apply them afterwards
        #
        add_temps = dict ( )
        for name, value in symbols.items ( ):
            #
            # unresolved symbols have 'None' as their value
            #
            if value is None:
                #
                # is this a stencil's attribute?
                #
                if 'self' in name:
                    attr  = name.split ('.')[1]
                    value = getattr (self, attr, None)

                    #
                    # NumPy arrays kept as stencil attributes are considered
                    # temporary data fields
                    #
                    if isinstance (value, np.ndarray):
                        #
                        # the new temporary data field will be added later,
                        # to prevent changes in the underlying data structure
                        # during the loop
                        #
                        add_temps[name] = value
                    else:
                        symbols.add_constant (name, 
                                              value)
            #
            # TODO some symbols are just aliases to other symbols
            #
            if isinstance (value, str):
                if value in symbols.keys ( ) and symbols[value] is not None:
                    #symbols[name] = symbols[value]
                    logging.warning ("Variable aliasing is not supported")

        #
        # update the symbol table now the loop has finished
        #
        for k,v in add_temps.items ( ):
            #
            # remove this field from the symbol table before adding it
            # as a temporary data field
            #
            del symbols[k]
            symbols.add_temporary (k, v)

        #
        # print the discovered symbols
        #
        logging.debug ("Symbols found using run-time information:")
        for k,v in self.inspector.symbols.items ( ):
            logging.debug ("\t%s:\t%s" % (k, str (v)))



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

        #
        # run the selected backend version
        #
        logging.info ("Running in %s mode ..." % self.backend.capitalize ( ))
        if self.backend == 'c++':
            #
            # try to resolve all symbols before compiling:
            # first with a static code analysis ...
            #
            self.inspector.analyze (**kwargs)
            
            #
            # ... then including run-time information
            #
            self._resolve (self.inspector.symbols)

            #
            # generate the code of all functors in this stencil
            #
            for func in self.inspector.functors:
                func.generate_code ( )

            #
            # compile the generated code
            #
            try:
                self.inspector.compile ( )
            except RuntimeError:
                logging.error ("Compilation failed")
            else:
                #
                # extract the buffer pointers from the parameters (NumPy arrays)
                #
                params = list (self.inspector.dimensions)
                functor_params = self.inspector.functors[0].params
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
        #
        # run in Python mode
        #
        elif self.backend == 'python':
            self.kernel (**kwargs)
        else:
            logging.warning ("Unknown backend [%s]" % self.backend)


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

    def __sub__ (self, other):
        raise NotImplementedError ("Offsets are not supported with '-'.")

