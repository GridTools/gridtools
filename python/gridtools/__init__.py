# -*- coding: utf-8 -*-
import sys
import ast
import logging
import warnings

import numpy as np

from gridtools.functor import StencilFunctor, FunctorBody, FunctorParameter



class StencilSymbols (object):
    """
    Keeps a repository of all symbols, e.g., constants, aliases, temporary
    data fields, functor parameters, appearing in a user-defined stencil.-
    """
    def __init__ (self):
        #
        # we categorize all symbols in these groups
        #
        self.groups = set (('_constant',    # a value, maybe available at runtime;
                            '_alias',       # variable aliasing;
                            '_temp_field',  # temporary data fields;
                           ))              # functor parameters are identified
                                           # by the functor's name and added later
        #
        # initialize the container (k=group, v=dict), where
        # dict contains k=symbol name, v=symbol value
        #
        self.symbol_table = dict ( )
        for g in self.groups:
            self.symbol_table[g] = dict ( )


    def __delitem__ (self, name):
        """
        Removes the symbol entry 'name' from this instance.-
        """
        for g in self.groups:
            if name in self.symbol_table[g].keys ( ):
                return self.symbol_table[g].__delitem__ (name)
        raise KeyError ("No symbol named '%s'" % name)


    def __getitem__ (self, name):
        """
        Returns the value of symbol 'name' or None if not found.-
        """
        for g in self.groups:
            if name in self.symbol_table[g].keys ( ):
                return self.symbol_table[g][name]
        return None
     

    def _add (self, name, value, group):
        """
        Adds the received symbol into the corresponding container:

            name    name of this symbol;
            value   value of this symbol;
            group   container group into which to add the symbol.-
        """
        if group in self.groups:
            if name in self.symbol_table[group].keys ( ):
                logging.info ("Updated value of symbol '%s'" % name)
            self.symbol_table[group][name] = value


    def add_alias (self, name, value):
        """
        Adds an alias to the stencil's symbols:

            name    name of this symbol;
            value   value of this symbol.-
        """
        logging.info ("Alias '%s' points to '%s'" % (name,
                                                     str (value)))
        self._add (name, str (value), '_alias')


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
        self._add (name, value, '_constant')


    def add_functor (self, name):
        """
        Returns a new dictionary for keeping the functor parameters:

            name    a unique name for the functor.-
        """
        if name in self.groups:
            raise NameError ("Functor '%s' already exists in symbol table.-" % name)
        else:
            self.groups.add (name)
            self.symbol_table[name] = dict ( )
            return self.symbol_table[name]


    def add_temporary (self, name, value):
        """
        Adds a temporary data field stencil's symbols:

            name    name of this symbol;
            value   value of this symbol (a NumPy array).-
        """
        if value is None:
            raise ValueError ("Value of temporary field '%s' is None" % name)
        elif isinstance (value, FunctorParameter):
            #
            # add the field as a temporary
            #
            self._add (name, value, '_temp_field')
            logging.info ("Temporary field '%s' has dimension %s" % (name,
                                                                     value.dim))
        else:
            raise TypeError ("Invalid type '%s' for temporary field '%s'" % 
                             (type (value), name))


    def arrange_ids (self):
        """
        Rearranges the IDs of all data fields in order to uniquely identify
        each of them.-
        """
        curr_id = 0
        for g in sorted (self.groups, reverse=True):
            for k,v in self.symbol_table[g].items ( ):
                try:
                    v.id = curr_id
                    curr_id += 1
                except AttributeError:
                    #
                    # not a data field, ignore it
                    #
                    pass


    def get_functor_params (self, funct_name):
        """
        Returns all parameters of functor 'funct_name'.-
        """
        if funct_name is not None:
            if funct_name in self.symbol_table.keys ( ):
                return self.symbol_table[funct_name].values ( )
            else:
                raise KeyError ("Functor '%s' is not a registered symbol in functor '%s'" % (name, funct_name))


    def is_parameter (self, name, funct_name=None):
        """
        Returns True is symbol 'name' is a functor parameter of 'funct_name'.
        If 'functor' is None, all registered functors are checked.-
        """
        if funct_name is not None:
            if funct_name in self.symbol_table.keys ( ):
                return name in self.symbol_table[funct_name].keys ( )
            else:
                raise KeyError ("Symbol '%s' is not registered in functor '%s'" % (name, funct_name))
        else:
            functors = [k for k in self.symbol_table.keys ( ) if not k.startswith ('_')]
            for f in functors:
                if name in self.symbol_table[f].keys ( ):
                    return True
            return False


    def is_temporary (self, name):
        """
        Returns True if symbol 'name' is a temporary data field.-
        """
        return name in self.symbol_table['_temp_field'].keys ( )


    def items (self):
        """
        Returns all symbols in as (key, value) pairs, sorted by key.-
        """
        import operator

        for g in self.groups:
            sorted_symbols = sorted (self.symbol_table[g].items ( ),
                                     key=operator.itemgetter (0),
                                     reverse=True)
            for k,v in sorted_symbols:
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
                            logging.warning ("Dimensions of parameter '%s':%s do not match %s" % (k, v.shape, self.dimensions))
                    else:
                        logging.warning ("Parameter '%s' is not a NumPy array" % k)
                else:
                    logging.warning ("Ignoring parameter '%s'" % k)


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
        #
        # initialize the template renderer environment
        #
        jinja_env = Environment (loader=PackageLoader ('gridtools',
                                                       'templates'))
        jinja_env.filters["join_with_prefix"] = join_with_prefix

        #
        # prepare the parameters used by the template engine
        #
        functor_source   = ""
        functor_template = jinja_env.get_template ("functor.h")
        temp_params = [v for k,v in self.symbols.items ( ) if self.symbols.is_temporary (k)]
        functor_params = list (self.symbols.get_functor_params (self.functors[0].name))
        all_params = temp_params + functor_params

        #
        # render the source code for each of the functors
        #
        for f in self.functors:
            functor_source += functor_template.render (functor=f,
                                                       all_params=all_params,
                                                       temp_params=temp_params,
                                                       functor_params=functor_params)

        #
        # instantiate each of the templates and render them
        #
        header = jinja_env.get_template ("stencil.h")
        cpp    = jinja_env.get_template ("stencil.cpp")
        make   = jinja_env.get_template ("Makefile")

        return (header.render (stencil=self,
                               functors=self.functors,
                               functor_source=functor_source,
                               all_params=all_params,
                               temp_params=temp_params,
                               functor_params=functor_params),
                cpp.render  (stencil=self,
                             functor_params=functor_params),
                make.render (stencil=self))


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
        self.lib_file  = path.join (tmp_dir, "lib%s.so" % self.name.lower ( ))
        self.hdr_file  = path.join (tmp_dir, '%s.h' % self.name)
        self.cpp_file  = path.join (tmp_dir, '%s.cpp' % self.name)
        self.make_file = path.join (tmp_dir, 'Makefile')

        #
        # ... and populate it
        #
        logging.info ("Compiling C++ code in [%s]" % tmp_dir)
        hdr_src, cpp_src, make_src = self.translate ( )

        with open (self.hdr_file, 'w') as hdr_hdl:
            hdr_hdl.write (hdr_src)

        with open (self.cpp_file, 'w') as cpp_hdl:
            cpp_hdl.write (cpp_src)

        with open (self.make_file, 'w') as make_hdl:
            make_hdl.write (make_src)

        #
        # before starting the compilation of the dynamic library
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
            self.lib_obj = cdll.LoadLibrary ("%s" % self.lib_file)
        except OSError:
            self.lib_obj = None
            raise RuntimeError ("Cannot load dynamically-compiled library")


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
                            # a name for this functor
                            #
                            funct_name = 'functor_%s' % ''.join ([choice (digits) for n in range (4)])
                            
                            #
                            # the functor parameters will be kept here
                            #
                            funct_params = self.symbols.add_functor (funct_name)

                            #
                            # create the functor object and analyze its code
                            #
                            funct = StencilFunctor (funct_name,
                                                    n,
                                                    funct_params,
                                                    self.symbols)
                            funct.analyze_params (node.args.args)
                            self.functors.append (funct)
                            #
                            # functor's body
                            #
                            funct.body = FunctorBody (n.body,
                                                      funct_params,
                                                      self.symbols)
                            logging.info ("Functor '%s' created" % funct.name)
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
        # update the symbol table now the loop has finished ...
        #
        for k,v in add_temps.items ( ):
            #
            # remove this field from the symbol table before adding it
            # as a temporary data field
            #
            del symbols[k]
            #
            # remove the 'self' from the attribute name
            #
            name = k.split ('.')[1]
            temp_field = FunctorParameter (name)
            temp_field.dim = v.shape
            symbols.add_temporary (name, temp_field)
        #
        # and rearrange all data fields IDs
        #
        symbols.arrange_ids ( )

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
            # automatic compilation only if the library is not available
            #
            if self.inspector.lib_obj is None:
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
                try:
                    for func in self.inspector.functors:
                        func.generate_code (self.inspector.src)

                except Exception as e:
                    logging.error ("Error while generating code %s" % str (e))
                    raise e

                else:
                    #
                    # compile the generated code
                    #
                    try:
                        self.inspector.compile ( )
                    except RuntimeError:
                        logging.error ("Compilation failed")
                        return
            #
            # call the compiled library
            #
            params = list (self.inspector.dimensions)
            #
            # extract the buffer pointers from the parameters (NumPy arrays)
            #
            functor_params = list (self.inspector.symbols.get_functor_params (self.inspector.functors[0].name))
            for p in functor_params:
                if p.name in kwargs.keys ( ):
                    params.append (kwargs[p.name].ctypes.data_as (ctypes.c_void_p))
                else:
                    logging.warning ("Missing parameter [%s]" % p.name)
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
