# -*- coding: utf-8 -*-
import sys
import ast
import compiler
import warnings

import numpy as np




class FunctorBody ( ):
    """
    Represents the Do( ) function of a stencil's functor. 
    It inherits the visitor pattern from compiler.visitor.ASTVisitor, to 
    recursively generate C++ code for each syntactical Python construct. 
    This code was adapted from the SHED SKIN Python-to-C++ Compiler, which is
    copyright 2005-2013 Mark Dufour; License GNU GPL version 3.-
    """
    def generate_code (self, gx):
        for module in gx.modules.values():
            if not module.builtin:
                gv = GenerateVisitor(gx, module)
                walk(module.ast, gv)
                gv.out.close()
                gv.header_file()
                gv.out.close()
                gv.insert_consts(declare=False)
                gv.insert_consts(declare=True)
                gv.insert_extras('.hpp')
                gv.insert_extras('.cpp')
        generate_makefile(gx)



class FunctorParameter ( ):
    """
    Represents a parameter of a stencil functor.-
    """
    def __init__ (self, name):
        """
        Creates a new parameter with the received name.-
        """
        self.id     = None
        self.name   = None 
        self.input  = None
        self.output = None
        self.set_name (name)

    def set_name (self, name):
        """
        Sets a new name to this functor parameter.-
        """
        if name.startswith ('in_'):
            #
            # functor input parameter
            #
            self.name = name
            self.input = True
            self.output = False
        elif name.startswith ('out_'):
            #
            # functor input parameter
            #
            self.name = name
            self.input = False
            self.output = True
        else:
            warnings.warn ("ignoring functor parameter [%s]\n" % name,
                           UserWarning)



class StencilFunctor ( ):
    """
    Represents a functor inside a multi-stage stencil.-
    """
    def __init__ (self, node=None):
        """
        Constructs a new StencilFunctor:

            node    the FunctionDef AST node (see
                    https://docs.python.org/3.4/library/ast.html) of the
                    Python function from which this functor will be built.-
        """
        #
        # a name to uniquely identify this functor
        #
        self.name = None
        #
        # a list to keep the functor parameters
        #
        self.params = list ( )
        #
        # the body of the functor is inlined from the 'for' loops
        #
        self.body = None
        #
        # the AST node of the Python function representing this functor
        #
        self.set_ast (node)


    def set_ast (self, node):
        """
        Speficies the AST describing the operations this functor should 
        implement:

            node    a FunctionDef AST node (see
                    https://docs.python.org/3.4/library/ast.html).-
        """
        self.node  = node 
        self.name  = "%s_functor" % node.name


    def analyze_params (self):
        """
        Extracts the parameters of the Python fuction before translating
        them to C++ functor code.-
        """
        try:
            #
            # analyze the function parameters
            #
            param_list = self.node.args.args
            for p in param_list:
                par = FunctorParameter (p.arg)
                #
                # the name is None if the parameter was ignored/invalid
                #
                if par.name is not None:
                    par.id = len (self.params)
                    self.params.append (par)

        except AttributeError:
            warnings.warn ("AST node not set or it is not a FuctionDef\n",
                           RuntimeWarning)


    def analyze_loops (self):
        """
        Looks for 'get_interior_points' comprehensions within the 
        Python function.-
        """
        #
        # the loops are part of the function body
        #
        function_body = self.node.body
        for node in function_body:
            if isinstance (node, ast.For):
                #
                # the iteration should call 'get_interior_points'
                #
                call = node.iter
                if (call.func.value.id == 'self' and 
                    call.func.attr == 'get_interior_points'):
                    self.body = node.body


    def translate (self):
        """
        Translates this functor to C++, using the gridtools interface.-
        """
        from jinja2 import Environment, PackageLoader

        jinja_env = Environment (loader = PackageLoader ('gridtools',
                                                         'templates'))
        tpl = jinja_env.get_template ("functor.c")
        print (tpl.render (functor=self))





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
            super ( ).__init__ ( )
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
                self.kernel_func.translate ( )
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


