# -*- coding: utf-8 -*-
import sys
import ast
import warnings

import numpy as np




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
    def __init__ (self):
        self.name = None
        #
        # a list to keep the functor parameters
        #
        self.params = list ( )


    def set_ast (self, node):
        """
        Speficies the AST describing the operations this functor should 
        implement:

            node    a FunctionDef AST node (see
                    https://docs.python.org/3.4/library/ast.html).-
        """
        try:
            self.name  = "%s_functor" % node.name
            param_list = node.args.args
            for p in param_list:
                par = FunctorParameter (p.arg)
                #
                # the name is None if the parameter is ignored/invalid
                #
                if par.name is not None:
                    par.id = len (self.params)
                    self.params.append (par)

        except AttributeError:
            raise Warning ("Node [%s] is not a FunctionDef\n" % node.name)


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

        super ( ).__init__ ( )
        self.src = getsource (cls)
        self.kernel_func = None


    def analyze (self):
        """
        Analyzes the source code of this stencil.-
        """
        module = ast.parse (self.src)
        self.visit (module)


    def visit_FunctionDef (self, node):
        """
        Looks for the 'kernel' function:

            node    a node from the AST.-
        """
        #
        # look for the 'kernel' function, which is the starting point 
        # of the stencil
        #
        if node.name == 'kernel':
            self.kernel_func = StencilFunctor ( )
            self.kernel_func.set_ast (node)
            self.kernel_func.translate ( )
            print ("\tbody: %s" % node.body)







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
        self.halo = (1, 1,  )


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

