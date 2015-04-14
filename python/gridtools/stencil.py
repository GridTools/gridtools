# -*- coding: utf-8 -*-
import sys
import ast
import logging
import warnings

import numpy as np
import networkx as nx

from gridtools.symbol  import StencilScope, Scope
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
        import inspect

        if (issubclass (obj.__class__, MultiStageStencil) or
            issubclass (obj.__class__, CombinedStencil)):
            super ( ).__init__ ( )

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
            # symbols gathered after analyzing the user's stencil are kept here
            #
            self.scope = StencilScope ( )

            #
            # a list of functors of this stenctil;
            # the kernel function is the entry functor of any stencil
            #
            self.functors = list ( )
        else:
            raise TypeError ("Class '%s' must extend 'MultiStageStencil'" % 
                             obj.__class__)


    def static_analysis (self):
        """
        Performs a static analysis of the source code of this stencil.-
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

    
    def visit_For (self, node):
        """
        Looks for 'get_interior_points' comprehensions.-
        """
        from random import choice
        from string import digits

        #
        # the iteration should call 'get_interior_points'
        #
        call = node.iter
        if (call.func.value.id == 'self' and 
            call.func.attr == 'get_interior_points'):
            #
            # a random name for this functor
            #
            funct_name = 'functor_%03d' % len (self.scope.functor_scopes)
            
            #
            # create a new scope for the symbols of this functor
            #
            functor_scope = self.scope.add_functor (funct_name)

            #
            # create a functor object
            #
            funct = Functor (funct_name,
                             node,
                             functor_scope,
                             self.scope)
            self.functors.append (funct)
            logging.debug ("Functor '%s' created" % funct.name)


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



class Stencil ( ):
    """
    A base stencil class.-
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
        # a halo descriptor - see 'set_halo' below
        #
        self.set_halo ( (0, 0, 0, 0) )

        #
        # define the execution order in 'k' dimension - see 'set_k_direction' below
        #
        self.set_k_direction ('forward')

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


    def __hash__ (self):
        return self.name.__hash__ ( )


    def __repr__ (self):
        return self.name


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
        except OSError as e:
            self.lib_obj = None
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


    def recompile (self):
        """
        Marks the currently compiled library as dirty, needing recompilation.-
        """
        #
        # clear the compiled library and the symbols inspector
        #
        self.lib_obj   = None
        self.inspector = StencilInspector (self)


    def run (self, *args, halo=None, k_direction=None, **kwargs):
        """
        Starts the execution of the stencil:

            halo            a tuple defining a 2D halo over the given 
                            'data_field'. See 'set_halo';
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward' or 'backward'.-
        """
        import ctypes
        import networkx as nx

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
        if self.backend == 'c++':
            #
            # automatic compilation only if the library is not available
            #
            if self.lib_obj is None:
                #
                # try to resolve all symbols before compiling:
                # first with doing a static code analysis, ...
                #
                self.inspector.static_analysis ( )

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
                    self.inspector.scope.dump ( )
                    for f in self.inspector.functors:
                        f.scope.dump ( )

                #
                # generate the code of *all* functors in this stencil
                # and build a data-dependency graph among *all* data fields
                #
                try:
                    for func in self.inspector.functors:
                        func.generate_code (self.inspector.src)
                        self.inspector.scope.add_dependencies (func.get_dependency_graph ( ).edges ( ))

                except Exception as e:
                    logging.error ("Error while generating code\n%s" % str (e))
                    raise e

                else:
                    #
                    # compile the generated code
                    #
                    try:
                        self.compile ( )
                    except Exception as e:
                        logging.error ("%s\nCompilation failed" % str(e))
                        return
            #
            # prepare the list of parameters to call the library function
            #
            lib_params = list (self.inspector.domain)

            #
            # extract the buffer pointers from the NumPy arrays
            #
            for p in self.inspector.scope.get_parameters ( ):
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

        params = list (self.inspector.scope.get_parameters ( ))
        temps  = list (self.inspector.scope.get_temporaries ( ))

        return (header.render (namespace=self.name.lower ( ),
                               stencil=self,
                               scope=self.inspector.scope,
                               params=params,
                               temps=temps,
                               params_temps=params + temps,
                               functor_src=functor_src,
                               functors=self.inspector.functors),
                cpp.render  (stencil=self,
                             params=params),
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
        #
        # a dictionary to keep track of the intermediate temporary fields
        #
        self.temporaries = dict ( )


    def _plot_graph (self, G):
        """
        Renders graph 'G' using 'matplotlib'.-
        """
        try:
            import matplotlib as plt

            pos = nx.spring_layout (G)
            nx.draw_networkx_nodes (G,
                                    pos=pos)
            nx.draw_networkx_edges (G,
                                    pos=pos,
                                    arrows=True)
            nx.draw_networkx_labels (G,
                                     pos=pos)
        except ImportError:
            logging.warning ("MatplotLib is not available")


    def _prepare_parameters (self, stencil, **kwargs):
        """
        Extracts the parameters for 'stencil' from 'kwargs', and create
        temporaries for linked stencils.-
        """
        domain    = None
        ret_value = dict ( )
        for p in stencil.scope.get_parameters ( ):
            try:
                if isinstance (kwargs[p.name], np.ndarray):
                    domain = kwargs[p.name].shape
                ret_value[p.name] = kwargs[p.name]
            except KeyError:
                #
                # get the value of this parameter from the data-dependency graph
                #
                if self.data_graph.has_node (p.name):
                    linked_data = self.data_graph.successors (p.name)
                    if len (linked_data) == 0:
                        #
                        # create a new temporary data field
                        #
                        self.temporaries[p.name] = None
                    elif len (linked_data) == 1:
                        #
                        # get the value from the linked field
                        #
                        ret_value[p.name] = self.temporaries[linked_data[0]]
                        try:
                            domain = ret_value[p.name].shape
                        except AttributeError:
                            #
                            # not a NumPy array ... no problem
                            #
                            pass
                else:
                    raise RuntimeError ("Parameter '%s' not found ... this shouldn't be happening ..."
                                        % p.name)
        #
        # generate a temporary for every missing parameter
        #
        if domain is not None:
            for k,v in self.temporaries.items ( ):
                if v is None:
                    self.temporaries[k] = np.zeros (domain)
                    ret_value[k]        = self.temporaries[k]
        else:
            raise RuntimeError ("Could not infer data-field dimension sizes")
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
                            for n in input_stencil.data_graph.nodes_iter ( ):
                                self.data_graph.add_node (n)
                            for u,v in input_stencil.data_graph.edges_iter ( ):
                                self.data_graph.add_edge (u, v)
                            #
                            #
                            # link the data dependency with the other stencil
                            #
                            linked_stencil        = input_stencil.get_root ( )
                            linked_stencil_output = input_stencil.execution_graph.node[linked_stencil]['output']
                            self.data_graph.add_node (input_param)
                            self.data_graph.add_node (linked_stencil_output)
                            self.data_graph.add_edge (input_param,
                                                      linked_stencil_output)
                            #
                            # ... and the execution order
                            #
                            self.execution_graph.add_edge (stencil, 
                                                           linked_stencil)
                        except AttributeError:
                            logging.error ("Parameter '%s' should hold an instance of CombinedStencil" % input_param)
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


    def run (self, *args, halo=None, k_direction=None, **kwargs):
        """
        Starts the execution of the stencil:

            halo            a tuple defining a 2D halo over the given 
                            'data_field'. See 'set_halo';
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward' or 'backward'.-
        """
        #
        # make sure all needed parameters were provided
        #
        for p in self.scope.get_parameters ( ):
            if p.name not in kwargs.keys ( ):
                raise ValueError ("Missing parameter '%s'" % p.name)

        #
        # run in Python mode
        #
        if self.backend == 'python':
            #
            # execution order is from the leafs towards the root
            #
            for st in nx.dfs_postorder_nodes (self.execution_graph,
                                              source=self.get_root ( )):
                params = self._prepare_parameters (st,
                                                   **kwargs)
                st.run (*args, 
                        halo=halo, 
                        k_direction=k_direction,
                        **params)
