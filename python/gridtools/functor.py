# -*- coding: utf-8 -*-
import sys
import ast
import warnings

import numpy as np

import gridtools.python as py



class FunctorBody (ast.NodeVisitor):
    """
    Represents the Do( ) function of a stencil's functor in AST form.-
    """
    def __init__ (self, nodes, params):
        """
        Constructs a functor body object using the received node:

            node    an AST-node list representing the body of this functor;
            params  the list of FunctorParameters of this functor.-
        """
        self.params = params
        #
        # initialize an empty C++ code string
        #
        self.cpp = ""
        try:
            if len (nodes) > 0:
                self.nodes = nodes
        except TypeError:
            warnings.warn ("FunctorBody expects a list of AST nodes.",
                           RuntimeWarning)


    def generate_code (self):
        """
        Generates C++ code from the AST backing this object.-
        """
        for n in self.nodes:
            self.cpp = self.visit (n)


    def visit_Subscript (self, node):
        """
        Generates code from Subscript node, i.e., expr[expr].-
        """
        #
        # check if subscripting any of the known data fields
        #
        val = node.value
        if isinstance (val, ast.Name):
            name = val.id
            if name in self.params:
                return "dom(%s( ))" % name
            else:
                return name
        #
        # TODO check if the subscript has shifting
        #
        return ''


    def visit_Assign (self, node):
        """
        Generates code from an Assignment node, i.e., expr = expr.-
        """
        if len (node.targets) < 2:
            return "%s = %s;" % (self.visit (node.targets[0]),  # lvalue
                                 self.visit (node.value))       # rvalue
        else:
            for tgt in node.targets:
                self.visit (tgt)




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

    def __eq__ (self, other):
        if isinstance (other, str):
            return self.name == other
        else:
            return (isinstance (other, self.__class__)
                    and self.name == other.name)

    def __ne__ (self, other):
        return not self.__eq__ (other)

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
            warnings.warn ("AST node not set or it is not a FunctionDef\n",
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
                    self.body = FunctorBody (node.body,
                                             self.params)
                    self.body.generate_code ( )

