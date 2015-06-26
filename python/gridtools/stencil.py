# -*- coding: utf-8 -*-
import ast
import logging

import numpy as np
import networkx as nx

from gridtools.utils   import Utilities
from gridtools.symbol  import StencilScope
from gridtools.functor import Functor




class InteriorPoint (tuple):
    """
    Represents the point within a NumPy array at the given coordinates.-
    """
    def __add__ (self, other):
        if len (self) != len (other):
            raise ValueError ("Points have different dimensions.")
        return tuple (map (sum, zip (self, other)))

    def __sub__ (self, other):
        raise NotImplementedError ("Offsets with '-' are not supported.")



class StencilInspector (ast.NodeVisitor):
    """
    Inspects the source code of a stencil definition using its AST.-
    """
    def __init__ (self, obj):
        """
        Creates an inspector object using the source code of a stencil:

            obj     a Stencil object.-
        """
        if (issubclass (obj.__class__, MultiStageStencil) or
            issubclass (obj.__class__, CombinedStencil)):
            super ( ).__init__ ( )

            #
            # a reference to the MultiStageStencil we have to inspect
            #
            self.stencil = obj
            #
            # stencil's source code
            #
            self.src = self._extract_source ( )
            #
            # the domain dimensions over which this stencil operates
            #
            self.domain  = None
            #
            # symbols gathered after analyzing the user's stencil are kept here
            #
            self.scope = StencilScope ( )
            #
            # a list of functors (i.e. stages) of this stenctil;
            # the kernel function is the entry functor of any stencil
            #
            self.functors     = list ( )
            self.functor_defs = list ( )
        else:
            raise TypeError ("Class '%s' must extend 'MultiStageStencil'" % 
                             obj.__class__)


    def _extract_source (self):
        """
        Extracts the source code from the user-defined stencil.-
        """
        import inspect

        src = 'class %s (%s):\n' % (str (self.stencil.__class__.__name__),
                                    str (self.stencil.__class__.__base__.__name__))
        #
        # first the constructor and stages
        #
        for (name,fun) in inspect.getmembers (self.stencil,
                                              predicate=inspect.ismethod):
            try:
                if name == '__init__' or name.startswith ('stage_'):
                    src += inspect.getsource (fun)
            except OSError:
                try:
                    #
                    # is this maybe a notebook session?
                    #
                    from IPython.code import oinspect
                    src += oinspect.getsource (fun)
                except Exception:
                    raise RuntimeError ("Could not extract source code from '%s'" 
                                        % self.stencil.__class__)
        #
        # then the kernel
        #
        for (name,fun) in inspect.getmembers (self.stencil,
                                              predicate=inspect.ismethod):
            try:
                if name == 'kernel':
                    src += inspect.getsource (fun)
            except OSError:
                try:
                    #
                    # is this maybe a notebook session?
                    #
                    from IPython.code import oinspect
                    src += oinspect.getsource (fun)
                except Exception:
                    raise RuntimeError ("Could not extract source code from '%s'" 
                                        % self.stencil.__class__)
        return src


    def static_analysis (self):
        """
        Performs a static analysis of the source code of this stencil.-
        """
        if self.src:
            #
            # do not the static analysis twice over the same code
            #
            if len (self.functors) > 0:
                self.__init__ (self.stencil)
            #
            # analysis starts by parsing the stencil source code
            #
            self.ast_root = ast.parse (self.src)
            self.visit (self.ast_root)
            if len (self.functors) == 0:
                raise NameError ("Could not extract any stencil stage")
        else:
            #
            # if the source code is not available, we may infer the user is
            # running from an interactive session
            #
            raise RuntimeError ("Please save your stencil classes to a file before changing the backend")
        #
        # print out the discovered symbols if in DEBUG mode
        #
        if __debug__:
            logging.debug ("Symbols found after static code analysis:")
            self.scope.dump ( )


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
                self.scope.add_parameter (n.arg,
                                                  read_only=read_only)


    def resolve (self):
        """
        Attempts to aquire more information about the discovered symbols
        with runtime information of user's stencil instance.-
        """
        for s in self.scope.get_all ( ):
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
                        self.scope.add_temporary (s.name,
                                                  s.value)
                    else:
                        self.scope.add_constant (s.name, 
                                                 s.value)


    def resolve_params (self, **kwargs):
        """
        Attempts to aquire more information about the discovered parameters 
        using runtime information.
        """
        for k,v in kwargs.items ( ):
            if self.scope.is_parameter (k):
                if isinstance (v, np.ndarray):
                    #
                    # update the value of this parameter
                    #
                    self.scope.add_parameter (k,
                                              v,
                                              read_only=self.scope[k].read_only)
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
                self.scope.add_constant (lvalue, rvalue)
                logging.debug ("Adding numeric constant '%s'" % lvalue)
            #
            # variable names are resolved using runtime information
            #
            elif isinstance (rvalue_node, ast.Name):
                try:
                    rvalue = eval (rvalue_node.id)
                    self.scope.add_constant (lvalue, rvalue)
                    logging.debug ("Adding constant '%s'" % lvalue)

                except NameError:
                    self.scope.add_constant (lvalue, None)
                    logging.debug ("Delayed resolution of constant '%s'" % lvalue)
            #
            # function calls are resolved later by name
            #
            elif isinstance (rvalue_node, ast.Call):
                rvalue = None
                self.scope.add_constant (lvalue, rvalue)
                logging.debug ("Constant '%s' holds a function value" % lvalue)
            #
            # attributes are resolved using runtime information
            #
            elif isinstance (rvalue_node, ast.Attribute):
                rvalue = getattr (eval (rvalue_node.value.id),
                                  rvalue_node.attr)
                self.scope.add_constant (lvalue, rvalue)
                logging.debug ("Constant '%s' holds an attribute value" % lvalue)
            #
            # try to discover the correct type using runtime information
            #
            else:
                #
                # we keep all other expressions and try to resolve them later
                #
                self.scope.add_constant (lvalue, None)
                logging.debug ("Constant '%s' will be resolved later" % lvalue)

    
    def visit_Expr (self, node):
        """
        Looks for named stages within a stencil.-
        """
        if isinstance (node.value, ast.Call):
            call = node.value
            if (isinstance (call.func, ast.Attribute) and
                isinstance (call.func.value, ast.Name)):
                if (call.func.value.id == 'self' and
                    call.func.attr.startswith ('stage_') ):
                    #
                    # found a new stage
                    #
                    funct_name  = '%s_%s_%03d' % (self.stencil.name.lower ( ),
                                                    call.func.attr,
                                                    len (self.scope.functor_scopes))
                    funct_scope = self.scope.add_functor (funct_name)
                    #
                    # extract its parameters
                    #
                    if len (call.args) > 0:
                        logging.warning ("Ignoring positional arguments when calling intermediate stages")
                    else:
                        for kw in call.keywords:
                            if isinstance (kw.value, ast.Attribute):
                                funct_scope.add_alias (kw.arg,
                                                       '%s.%s' % (kw.value.value.id,
                                                                  kw.value.attr))
                            elif isinstance (kw.value, ast.Name):
                                funct_scope.add_alias (kw.arg,
                                                       kw.value.id)
                            else:
                                raise TypeError ("Unknown type '%s' of keyword argument '%s'" 
                                                 % (kw.value.__class__, kw.arg))
                    #
                    # look for its definition
                    #
                    for fun_def in self.functor_defs:
                        if fun_def.name == call.func.attr:
                            for node in fun_def.body:
                                if isinstance (node, ast.For):
                                    self.visit_For (node,
                                                    funct_name=funct_name,
                                                    funct_scope=funct_scope,
                                                    independent=True)


    def visit_For (self, node, funct_name=None, funct_scope=None, independent=False):
        """
        Looks for 'get_interior_points' comprehensions.-
        """
        #
        # the iteration should call 'get_interior_points'
        #
        call = node.iter
        if (call.func.value.id == 'self' and 
            call.func.attr == 'get_interior_points'):
            #
            # a random name for this functor if none given
            #
            if funct_name is None and funct_scope is None:
                funct_name  = '%s_functor_%03d' % (self.stencil.name.lower ( ),
                                                   len (self.scope.functor_scopes))
                funct_scope = self.scope.add_functor (funct_name)
            #
            # create a functor object
            #
            funct = Functor (funct_name,
                             node,
                             funct_scope,
                             self.scope)
            funct.independent = independent
            self.functors.append (funct)
            logging.info ("Stage '%s' created" % funct.name)


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
            docstring = ast.get_docstring(node)
            #
            # should be a call to the parent-class constructor
            #
            pcix = 0 # Index, amongst the children nodes, of the call to parent constructor
            for n in node.body:
                if isinstance(n.value, ast.Str):
                    # Allow for the docstring to appear before the call to the parent constructor
                    if n.value.s.lstrip() != docstring:
                        pcix = pcix + 1
                       
                else:
                    pcix = pcix + 1
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
            if pcix != 1:
                raise ReferenceError ("Parent constructor is NOT the first operation of the child constructor")
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
            if node.returns is not None:
                raise ValueError ("The 'kernel' function should return 'None'")
            #
            # the parameters of the 'kernel' function are the stencil
            # arguments in the generated code
            #
            self.analyze_params (node.args.args)

            #
            # continue traversing the AST
            #
            for n in node.body:
                self.visit (n)
        #
        # other function definitions are save for potential use later
        #
        else:
            self.functor_defs.append (node)



class Stencil ( ):
    """
    A base stencil class.-
    """
    #
    # a utilities class shared by all stencils
    #
    utils = Utilities ( )

    def __init__ (self):
        #
        # a unique name for the stencil object
        #
        self.name = self.__class__.__name__.capitalize ( )
        #
        # the inspector object is used to JIT-compile this stencil
        #
        self.inspector = StencilInspector (self)
        #
        # these entities are automatically generated at compile time
        #
        self.src_dir      = None
        self.lib_file     = None
        self.cpp_file     = None
        self.make_file    = None
        self.fun_hdr_file = None
        #
        # a reference to the compiled dynamic library
        #
        self.lib_obj = None
        #
        # defines the way to execute the stencil, one of 'python' or 'c++'
        #
        self.backend = "python"
        #
        # a halo descriptor - see 'set_halo' below
        #
        self.set_halo ( (0, 0, 0, 0) )
        #
        # define the execution order in 'k' dimension - see 'set_k_direction' below
        #
        self.set_k_direction ('forward')


    def __hash__ (self):
        return self.name.__hash__ ( )


    def __repr__ (self):
        return self.name


    def _plot_graph (self, G):
        """
        Renders graph 'G' using 'matplotlib'.-
        """
        from gridtools import plt

        pos = nx.spring_layout (G)
        nx.draw_networkx_nodes (G,
                                pos=pos)
        nx.draw_networkx_edges (G,
                                pos=pos,
                                arrows=True)
        nx.draw_networkx_labels (G,
                                 pos=pos)
    
    
    def compile (self):
        """
        Compiles the translated code to a shared library, ready to be used.-
        """
        from os         import path, getcwd, chdir
        from numpy      import ctypeslib
        from subprocess import check_call

        try:
            #
            # start the compilation of the dynamic library
            #
            current_dir = getcwd ( )
            chdir (self.src_dir)
            check_call (["make", 
                         "--silent", 
                         "--file=%s" % self.make_file])
            chdir (current_dir)
            #
            # attach the library object
            #
            self.lib_obj = ctypeslib.load_library (self.lib_file,
                                                   self.src_dir)
        except Exception as e:
            logging.error ("Compilation error")
            self.lib_obj = None
            raise e


    def generate_code (self, src_dir=None):
        """
        Generates native code for this stencil:

            src_dir     directory where the files should be saved (optional).-
        """
        from os        import write, path, makedirs
        from tempfile  import mkdtemp
        from gridtools import JinjaEnv

        try:
            #
            # create directory and files for the generated code
            #
            if src_dir is None:
                self.src_dir = mkdtemp (prefix="__gridtools_")
            else:
                if not path.exists (src_dir):
                    makedirs (src_dir)
                self.src_dir = src_dir

            if self.backend == 'c++':
                extension = 'cpp'
            elif self.backend == 'cuda':
                extension = 'cu'
            else:
                raise RuntimeError ("Unknown backend '%s' in while generating code" % self.backend)
            self.cpp_file     = '%s.%s'    % (self.name, extension)
            self.lib_file     = 'lib%s'    % self.name.lower ( )
            self.make_file    = 'Makefile'
            self.fun_hdr_file = '%sFunctors.h' % self.name

            #
            # ... and populate them
            #
            logging.info ("Generating %s code in '%s'" % (self.backend.upper ( ),
                                                          self.src_dir))
            #
            # generate the code of *all* functors in this stencil,
            # build a data-dependency graph among *all* data fields
            #
            for func in self.inspector.functors:
                func.generate_code (self.inspector.src)
                self.scope.add_dependencies (func.get_dependency_graph ( ).edges ( ))
            fun_src, cpp_src, make_src = self.translate ( )

            with open (path.join (self.src_dir, self.fun_hdr_file), 'w') as fun_hdl:
                functors  = JinjaEnv.get_template ("functors.h")
                fun_hdl.write (functors.render (functor_src=fun_src))
            with open (path.join (self.src_dir, self.cpp_file), 'w') as cpp_hdl:
                cpp_hdl.write (cpp_src)
            with open (path.join (self.src_dir, self.make_file), 'w') as make_hdl:
                make_hdl.write (make_src)

        except Exception as e:
            logging.error ("Error while generating code:\n\t%s" % str (e))
            raise e


    def get_interior_points (self, data_field):
        """
        Returns an iterator over the 'data_field' without including the halo:

            data_field      a NumPy array;
        """
        try:
            if len (data_field.shape) != 3:
                raise ValueError ("Only 3D arrays are supported.")

        except AttributeError:
            raise TypeError ("Calling 'get_interior_points' without a NumPy array")
        else:
            #
            # calculate 'i','j' iteration boundaries based on 'halo'
            #
            i_dim, j_dim, k_dim = data_field.shape

            start_i = 0 + self.halo[1]
            end_i   = i_dim - self.halo[0]
            start_j = 0 + self.halo[3]
            end_j   = j_dim - self.halo[2]

            #
            # calculate 'k' iteration boundaries based 'k_direction'
            #
            if self.k_direction == 'forward':
                start_k = 0
                end_k   = k_dim
                inc_k   = 1
            elif self.k_direction == 'backward':
                start_k = k_dim - 1
                end_k   = -1
                inc_k   = -1
            else:
                logging.warning ("Ignoring unknown direction '%s'" % self.k_direction)

            #
            # return the coordinate tuples in the correct order
            #
            for i in range (start_i, end_i):
                for j in range (start_j, end_j):
                    for k in range (start_k, end_k, inc_k):
                        yield InteriorPoint ((i, j, k))


    @property
    def scope (self):
        return self.inspector.scope


    def kernel (self, *args, **kwargs):
        """
        This function is the entry point of the stencil and 
        should be implemented by the user.-
        """
        raise NotImplementedError ( )


    def plot_3d (self, Z):
        """
        Plots the Z field in 3D, returning a Matplotlib's Line3DCollection.-
        """
        if len (Z.shape) == 2:
            from gridtools import plt

            fig = plt.figure ( )
            ax  = fig.add_subplot (111,
                                   projection='3d',
                                   autoscale_on=True)
            X, Y = np.meshgrid (np.arange (Z.shape[0]),
                                np.arange (Z.shape[1]))
            im = ax.plot_wireframe (X, Y, Z,
                                    linewidth=1)
            return im
        else:
            logging.error ("The passed Z field should be 2D")


    def plot_dependency_graph (self):
        """
        Renders the data depencency graph using 'matplotlib'.-
        """
        self._plot_graph (self.scope.depency_graph)


    def recompile (self):
        """
        Marks the currently compiled library as dirty, needing recompilation.-
        """
        import _ctypes

        #
        # this only works in POSIX systems ...
        #
        if self.lib_obj is not None:
            _ctypes.dlclose (self.lib_obj._handle)
            del self.lib_obj
            self.lib_obj   = None
            self.inspector = StencilInspector (self)


    def resolve (self, **kwargs):
        """
        Resolve the names and types of the symbols used in this stencil.-
        """
        #
        # try to resolve all symbols by applying static-code analysis
        #
        self.inspector.static_analysis ( )

        #
        # ... and by including runtime information
        #
        self.inspector.resolve ( )
        self.inspector.resolve_params (**kwargs)
        
        #
        # print out the discovered symbols if in DEBUG mode
        #
        if __debug__:
            logging.debug ("Symbols found after using run-time resolution:")
            self.scope.dump ( )
            for f in self.inspector.functors:
                f.scope.dump ( )


    def run (self, *args, halo=None, k_direction=None, **kwargs):
        """
        Starts the execution of the stencil:

            halo            a tuple defining a 2D halo over the given parameters.
                            See 'set_halo';
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward' or 'backward'.-
        """
        import ctypes

        #
        # we only accept keyword arguments to avoid confusion
        #
        if len (args) > 0:
            raise KeyError ("Only keyword arguments are accepted")
        #
        # set halo and execution order in 'k' direction
        #
        self.set_halo        (halo)
        self.set_k_direction (k_direction)

        #
        # run the selected backend version
        #
        logging.info ("Running in %s mode ..." % self.backend.capitalize ( ))
        if self.backend == 'c++' or self.backend == 'cuda':
            #
            # automatic compilation only if the library is not available
            #
            if self.lib_obj is None:
                #
                # floating point precision validation
                #
                for key in kwargs:
                      if isinstance(kwargs[key], np.ndarray):
                          if not Stencil.utils.is_valid_float_type_size (kwargs[key]):
                              raise TypeError ("Element size of '%s' does not match that of the C++ backend."
                                               % key)
                          if self.backend == 'cuda' and not Stencil.utils.is_fortran_array_layout (kwargs[key]):
                              logging.info('Detected an incorrect array layout.  Checking if it can be converted.')
                              if Stencil.util.is_c_array_layout(kwargs[key]):
                                  logging.info('Converting array layout.')
                                  kwargs[key] = np.asfortranarray(kwargs[key])        # Attempt to reformat
                                  Stencil.utils.is_fortran_array_layout (kwargs[key]) # re-check
                              else
                                  raise TypeError ("An array does not have the correct layout for the cuda backend!")
                self.resolve (**kwargs)
                self.generate_code ( )
                self.compile ( )
            #
            # prepare the list of parameters to call the library function
            #
            lib_params = list (self.inspector.domain)

            #
            # extract the buffer pointers from the NumPy arrays
            #
            for p in self.scope.get_parameters ( ):
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
            logging.error ("Unknown backend '%s'" % self.backend)


    def set_halo (self, halo=(0,0,0,0)):
        """
        Applies the received 'halo' setting, which is defined as

            (halo in negative direction over _i_, 
             halo in positive direction over _i_,
             halo in negative direction over _j_,
             halo in positive direction over _j_).-
        """
        if halo is None:
            return
        if len (halo) == 4:
            if halo[0] >= 0 and halo[2] >= 0:
                if halo[1] >= 0 and halo[3] >= 0:
                    self.halo = halo
                    self.recompile ( )
                    logging.debug ("Setting halo to %s" % str (self.halo))
                else:
                    raise ValueError ("Invalid halo %s: definition for the positive halo should be zero or a positive integer" % str (halo))
            else:
                raise ValueError ("Invalid halo %s: definition for the negative halo should be zero or a positive integer" % str (halo))
        else:
            raise ValueError ("Invalid halo %s: it should contain four values" % str (halo))


    def set_k_direction (self, direction="forward"):
        """
        Applies the execution order in `k` dimension:

            direction   defines the execution order, which may be any of: 
                        forward or backward.-
        """
        accepted_directions = ["forward", "backward"]

        if direction is None:
            return

        if direction in accepted_directions:
            self.k_direction = direction
            self.recompile ( )
            logging.debug ("Setting k_direction to '%s'" % self.k_direction)
        else:
            logging.warning ("Ignoring unknown direction '%s'" % direction)


    def translate (self):
        """
        Translates this stencil to C++, using the gridtools interface, returning
        a string tuple of rendered files (functors, cpp, make).-
        """
        from gridtools import JinjaEnv

        #
        # render the source code for each of the functors
        #
        functor_src = ""
        for f in self.inspector.functors:
            functor_src += f.translate ( )
        #
        # instantiate each of the templates and render them
        #
        cpp    = JinjaEnv.get_template ("stencil.cpp")
        make   = JinjaEnv.get_template ("Makefile.%s" % self.backend)

        params = list (self.scope.get_parameters ( ))
        temps  = list (self.scope.get_temporaries ( ))

        functs     = dict ( )
        ind_functs = dict ( )

        functs[self.name]     = [f for f in self.inspector.functors if not f.independent]
        ind_functs[self.name] = [f for f in self.inspector.functors if f.independent]
       
        #
        # make sure there is at least one non-independent functor
        #
        if len (functs[self.name]) == 0:
            functs[self.name]     = [ self.inspector.functors[-1] ]
            ind_functs[self.name] = ind_functs[self.name][:-1]

        return (functor_src,
                cpp.render (fun_hdr_file         = self.fun_hdr_file,
                            stencil_name         = self.name,
                            stencils             = [self],
                            scope                = self.scope,
                            params               = params,
                            temps                = temps,
                            params_temps         = params + temps,
                            functors             = functs,
                            independent_functors = ind_functs),
                make.render (stencil=self))



class MultiStageStencil (Stencil):
    """
    A involving several stages, implemented as for-comprehensions.
    All user-defined stencils should inherit for this class.-
    """
    def __init__ (self):
        super ( ).__init__ ( )


    def build (self, output, **kwargs):
        """
        Use this stencil as part of a new CombinedStencil, the output parameter
        of which is called 'output' and should be linked to '**kwargs' parameters
        of the following stencil.-
        """
        #
        # make sure the output parameter is there
        #
        if output is None:
            raise ValueError ("You must specify an 'output' parameter")
        #
        # the keyword arguments map the output of this stencil
        # with an input parameter of the following one
        #
        ret_value = CombinedStencil ( )
        ret_value.add_stencil (self,
                               output=output,
                               **kwargs)
        return ret_value



class CombinedStencil (Stencil):
    """
    A stencil created as a combination of several MultiStageStencils.-
    """
    def __init__ (self):
        """
        Creates a CombinedStencil with an output field named 'output_field'.-
        """
        super ( ).__init__ ( )
        #
        # graphs for data-dependency and execution order
        #
        self.data_graph      = nx.DiGraph ( )
        self.execution_graph = nx.DiGraph ( )


    def _extract_domain (self, **kwargs):
        """
        Returns the current domain dimensions from 'kwargs'.-
        """
        domain = None
        for k,v in kwargs.items ( ):
            try:
                if domain is None:
                    domain = v.shape
                if domain != v.shape:
                    logging.warning ("Different domain sizes detected")
            except AttributeError:
                #
                # not a NumPy array
                #
                pass
        if domain is None:
            raise RuntimeError ("Could not infer data-field dimensions")
        return domain 


    def _prepare_parameters (self, stencil, **kwargs):
        """
        Extracts the parameters for 'stencil' from 'kwargs', and create
        intermediate data fields for linked stencils.-
        """
        ret_value = dict ( )
        domain    = self._extract_domain (**kwargs)
        for p in stencil.scope.get_parameters ( ):
            try:
                ret_value[p.name] = kwargs[p.name]
            except KeyError:
                #
                # get the value of this parameter from the data-dependency graph
                #
                if self.data_graph.has_node (p.name):
                    linked_data = self.data_graph.successors (p.name)
                    if len (linked_data) == 0:
                        #
                        # create a new linked data field
                        #
                        self.data_graph.node[p.name]['value'] = np.zeros (domain)
                        ret_value[p.name] = self.data_graph.node[p.name]['value']
                    elif len (linked_data) == 1:
                        #
                        # get the value from the linked field
                        #
                        ret_value[p.name] = self.data_graph.node[linked_data[0]]['value']
                else:
                    raise RuntimeError ("Parameter '%s' not found in data graph"
                                        % p.name)
        return ret_value


    def _update_parameters (self):
        """
        Updates the parameters of this combined stencil.-
        """
        #
        # add all the parameters of all the stencils combined
        #
        for stencil in self.execution_graph.nodes_iter ( ):
            for p in stencil.scope.get_parameters ( ):
                self.scope.add_parameter (p.name,
                                          p.value,
                                          p.read_only)
        #
        # remove linked parameters, i.e., output of one stencil is the input
        # of the following one
        #
        for n in self.data_graph.nodes_iter ( ):
            try:
                self.scope.remove (n)
            except KeyError:
                pass


    def add_stencil (self, stencil, output, **kwargs):
        """
        Adds a stencil, the output of which is called 'output' and should be
        forwarded into an input parameter of an adjacent stencil in the 
        execution graph.-
        """
        stencil.inspector.static_analysis ( )
        try:
            out_param = stencil.scope[output]
            if stencil.scope.is_parameter (out_param.name):
                #
                # add 'stencil' to the execution graph
                #
                stencil_params = [p for p in stencil.scope.get_parameters ( )]
                self.execution_graph.add_node (stencil,
                                               output=out_param.name,
                                               params=stencil_params)
                if len (kwargs) > 0:
                    #
                    # 'stencil' is linked to the output of other stencils
                    #
                    for input_param in kwargs.keys ( ):
                        input_stencil = kwargs[input_param]
                        try:
                            #
                            # update the execution graph ...
                            #
                            for n,d in input_stencil.execution_graph.nodes_iter (data=True):
                                self.execution_graph.add_node (n, d)
                            for u,v in input_stencil.execution_graph.edges_iter ( ):
                                self.execution_graph.add_edge (u, v)
                            #
                            # ... and the data graph of this stencil
                            #
                            for n,d in input_stencil.data_graph.nodes_iter (data=True):
                                self.data_graph.add_node (n, d)
                            for u,v in input_stencil.data_graph.edges_iter ( ):
                                self.data_graph.add_edge (u, v)
                            #
                            #
                            # link the data dependency with the other stencil
                            #
                            linked_stencil        = input_stencil.get_root ( )
                            linked_stencil_output = input_stencil.execution_graph.node[linked_stencil]['output']
                            self.data_graph.add_node (input_param, 
                                                      {'value': None})
                            self.data_graph.add_node (linked_stencil_output,
                                                      {'value': None})
                            self.data_graph.add_edge (input_param,
                                                      linked_stencil_output)
                            #
                            # ... and the execution dependency
                            #
                            self.execution_graph.add_edge (stencil, 
                                                           linked_stencil)
                        except AttributeError:
                            logging.error ("Parameter '%s' should hold an instance of '%s'"
                                           % (input_param, self.__class__))
                            return
                #
                # update the parameters of this combined stencil
                #
                self._update_parameters ( )
                logging.debug ("Execution graph of '%s'\n\t%s" % (self,
                                                                  self.execution_graph.edges (data=True)))
            else:
                raise ValueError ("'%s' is not a parameter of '%s'" % (output,
                                                                       stencil.name))
        except KeyError:
            raise ValueError ("'%s' is not a parameter of '%s'" % (output,
                                                                   stencil.name))


    def generate_code (self, src_dir=None):
        """
        Generates native code for this stencil:

            src_dir     directory where the files should be saved (optional).-
        """
        from os        import write, path, makedirs
        from tempfile  import mkdtemp
        from gridtools import JinjaEnv

        #
        # create directory and files for the generated code
        #
        if src_dir is None:
            self.src_dir = mkdtemp (prefix="__gridtools_")
        else:
            if not path.exists (src_dir):
                makedirs (src_dir)
            self.src_dir = src_dir

        functor_src       = ''
        self.lib_file     = 'lib%s' % self.name.lower ( )
        self.cpp_file     = '%s.cpp' % self.name
        self.make_file    = 'Makefile'
        self.fun_hdr_file = '%sFunctors.h' % self.name

        logging.info ("Generating C++ code in '%s'" % self.src_dir)
        #
        # generate the code for *all* functors within the combined stencil
        #
        for st in nx.dfs_postorder_nodes (self.execution_graph,
                                          source=self.get_root ( )):
            for func in st.inspector.functors:
                func.generate_code (st.inspector.src)
                st.scope.add_dependencies (func.get_dependency_graph ( ).edges ( ))
            fun_src, _, _  = st.translate ( )
            functor_src   += fun_src

        with open (path.join (self.src_dir, self.fun_hdr_file), 'w') as fun_hdl:
            functors = JinjaEnv.get_template ("functors.h")
            fun_hdl.write (functors.render (functor_src=functor_src))
        #
        # code for the stencil, the library entry point and makefile
        #
        cpp_src, make_src = self.translate ( )
        with open (path.join (self.src_dir, self.cpp_file), 'w') as cpp_hdl:
            cpp_hdl.write (cpp_src)
        with open (path.join (self.src_dir, self.make_file), 'w') as make_hdl:
            make_hdl.write (make_src)


    def get_root (self):
        """
        Returns the root node of the execution graph, i.e., the *last* stencil
        to be executed, including its associated data.-
        """
        ret_value = None
        for node in self.execution_graph.nodes_iter ( ):
            if len (self.execution_graph.predecessors (node)) == 0:
                if ret_value is None:
                    ret_value = node
                else:
                    raise RuntimeError ("The execution graph of '%s' contains two roots"
                                        % self)
        if ret_value is None:
            logging.warning ("Root node could not be found")
            ret_value = self.execution_graph.nodes ( )[0]
        return ret_value
   

    def plot_data_graph (self):
        """
        Renders the data graph using 'matplotlib'.-
        """
        self._plot_graph (self.data_graph)


    def plot_execution_graph (self):
        """
        Renders the execution graph using 'matplotlib'.-
        """
        self._plot_graph (self.execution_graph)


    def resolve (self, **kwargs):
        #
        # resolution order is from the leafs towards the root
        #
        for st in nx.dfs_postorder_nodes (self.execution_graph,
                                          source=self.get_root ( )):
            st.backend = self.backend
            params     = self._prepare_parameters (st,
                                                   **kwargs)
            st.resolve (**params)
            #
            # update the current field domain
            #
            if self.inspector.domain is not None:
                if self.inspector.domain != st.inspector.domain:
                    logging.warning ("Different domain sizes detected")
            self.inspector.domain = st.inspector.domain


    def run (self, *args, halo=None, k_direction=None, **kwargs):
        """
        Starts the execution of the stencil:

            halo            a tuple defining a 2D halo over the given parameters;
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward' or 'backward'.-
        """
        import ctypes

        #
        # make sure all needed parameters were provided
        #
        for p in self.scope.get_parameters ( ):
            if p.name not in kwargs.keys ( ):
                raise ValueError ("Missing parameter '%s'" % p.name)
        #
        # generate one compilation unit for all stencils combined
        #
        if self.backend == 'c++':
            #
            # automatic compilation only if the library is not available
            #
            if self.lib_obj is None:
                self.resolve (**kwargs)
                #
                # add extra parameters needed to connect linked stencils
                #
                for n,d in self.data_graph.nodes_iter (data=True):
                    succ = self.data_graph.successors (n)
                    if len (succ) == 0:
                        self.scope.add_parameter (n,
                                                  value=d['value'],
                                                  read_only=False)
                    elif len (succ) == 1:
                        self.scope.add_parameter (n,
                                                  value=self.data_graph.node[succ[0]]['value'],
                                                  read_only=True)
                #
                # continue generating the code ...
                #
                self.generate_code ( )
                self.compile ( )
            #
            # prepare the list of parameters to call the library function
            #
            lib_params = list (self.inspector.domain)

            #
            # extract the buffer pointers from the NumPy arrays
            #
            for p in self.scope.get_parameters ( ):
                if p.name in kwargs.keys ( ):
                    lib_params.append (kwargs[p.name].ctypes.data_as (ctypes.c_void_p))
                else:
                    logging.debug ("Adding linked parameter '%s'" % p.name)
                    linked_param = self.data_graph.node[p.name]['value']
                    if linked_param is None:
                        for s in self.data_graph.successors (p.name):
                            if self.data_graph.node[s]['value'] is not None:
                                linked_param = self.data_graph.node[s]['value']
                    lib_params.append (linked_param.ctypes.data_as (ctypes.c_void_p))
            #
            # call the compiled stencil
            # 
            self.lib_obj.run (*lib_params)
        elif self.backend == 'python':
            #
            # execution order is from the leafs towards the root
            #
            for st in nx.dfs_postorder_nodes (self.execution_graph,
                                              source=self.get_root ( )):
                st.backend = self.backend
                params     = self._prepare_parameters (st,
                                                       **kwargs)
                st.run (*args, 
                        halo=halo, 
                        k_direction=k_direction,
                        **params)


    def translate (self):
        """
        Translates this stencil to C++, using the gridtools interface, returning
        a string tuple of rendered files (cpp, make).-
        """
        from gridtools import JinjaEnv

        #
        # instantiate each of the templates and render them
        #
        cpp    = JinjaEnv.get_template ("stencil.cpp")
        make   = JinjaEnv.get_template ("Makefile.%s" % self.backend)

        params = list (self.scope.get_parameters ( ))
        temps  = list (self.scope.get_temporaries ( ))

        #
        # stencil list and functor dictionaries in execution order
        #
        stencils             = list ( )
        functors             = dict ( )
        independent_functors = dict ( )
        for st in nx.dfs_postorder_nodes (self.execution_graph,
                                          source=self.get_root ( )):
            if st in stencils:
                logging.warning ("Ignoring mutiple instances of stencil '%s' in '%s'"
                                 % (st, self))
            else:
                stencils.append (st)
            functors[st.name]             = list ( )
            independent_functors[st.name] = list ( )
            for f in st.inspector.functors:
                if f in functors[st.name] or f in independent_functors[st.name]:
                    logging.warning ("Ignoring mutiple instances of functor '%s' in stencil '%s'"
                                     % (f, st))
                else:
                    if f.independent:
                        independent_functors[st.name].append (f)
                    else:
                        functors[st.name].append (f)
        #
        # make sure there is at least one non-independent functor in each stencil
        #
        for st in stencils:
            if len (functors[st.name]) == 0:
                functors[st.name]             = [ st.inspector.functors[-1] ]
                independent_functors[st.name] = independent_functors[st.name][:-1]

        return (cpp.render (fun_hdr_file         = self.fun_hdr_file,
                            stencil_name         = self.name.lower ( ),
                            stencils             = stencils,
                            scope                = self.scope,
                            params               = params,
                            temps                = temps,
                            params_temps         = params + temps,
                            functors             = functors,
                            independent_functors = independent_functors),
                make.render (stencil=self))

