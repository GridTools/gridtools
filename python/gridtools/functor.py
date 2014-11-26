# -*- coding: utf-8 -*-
import sys
import ast
import logging
import warnings

import numpy as np



class FunctorBody (ast.NodeVisitor):
    """
    Represents the Do( ) function of a stencil's functor in AST form.-
    """
    def __init__ (self, nodes, params, symbols):
        """
        Constructs a functor body object using the received node:

            node    an AST-node list representing the body of this functor;
            params  a dict of FunctorParameters of this functor;
            symbols a dict of symbols available within the stencil where
                    the functor lives.-
        """
        self.params  = params
        self.symbols = symbols
        #
        # initialize an empty C++ code string
        #
        self.cpp = ''
        try:
            if len (nodes) > 0:
                self.nodes = nodes
        except TypeError:
            warnings.warn ("FunctorBody expects a list of AST nodes.",
                           RuntimeWarning)

    def _sign_operator (self, operation):
        """
        Returns the sign of an operation.-
        """
        if isinstance (operation, ast.Add):
            sign = '+'
        elif isinstance (operation, ast.Sub):
            sign = '-'
        elif isinstance (operation, ast.Mult):
            sign = '*'
        elif isinstance (operation, ast.Div):
            sign = '/'
        elif isinstance (operation, ast.Pow):
            sign = '^'
        else:
            sign = None
            logging.warning ("Cannot translate '%s'" % str (operation))
        return sign

         
    def generate_code (self):
        """
        Generates C++ code from the AST backing this object.-
        """
        for n in self.nodes:
            self.cpp += "%s;\n\t" % self.visit (n)


    def visit_Assign (self, node):
        """
        Generates code from an Assignment node, i.e., expr = expr.-
        """
        if len (node.targets) < 2:
            return "%s = %s" % (self.visit (node.targets[0]),  # lvalue
                                self.visit (node.value))       # rvalue
        else:
            for tgt in node.targets:
                self.visit (tgt)


    def visit_AugAssign (self, node):
        """
        Generates code for an operation-assignment node, i.e., expr += expr.-
        """
        sign = self._sign_operator (node.op)
        return "%s %s= %s" % (self.visit (node.target),
                              sign,
                              self.visit (node.value))


    def visit_BinOp (self, node):
        """
        Generates code for a binary operation, e.g., +,-,*, ...
        """
        sign = self._sign_operator (node.op)
        #
        # take care of the parenthesis for correct operation precedence
        #
        operand = []
        for op in [node.left, node.right]:
            if (isinstance (op, ast.Num) or 
                isinstance (op, ast.Name) or
                isinstance (op, ast.Attribute) or
                isinstance (op, ast.Subscript)):
                operand.append ('%s' % self.visit (op))
            else:
                operand.append ('(%s)' % self.visit (op))

        return "%s %s %s" % (operand[0], sign, operand[1])


    def visit_Num (self, node):
        """
        Returns the number in this node.-
        """
        return str(node.n)


    def visit_Subscript (self, node):
        """
        Generates code from Subscript node, i.e., expr[expr].-
        """
        if isinstance (node.slice, ast.Index):
            if isinstance (node.slice.value, ast.BinOp):
                #
                # this subscript has shifting
                #
                if isinstance (node.slice.value.op, ast.Add):
                    indexing = '(%s)' % ','.join ([str(e.n) for e in node.slice.value.right.elts])
                else:
                    indexing = ''
                    logging.warning ("Subscript shifting only supported with '+'")
            elif isinstance (node.slice.value, ast.Name):
                #
                # understand subscripting over 'get_interior_points'
                #
                if node.slice.value.id == 'p':
                    indexing = '( )'
                else:
                    indexing = ''
                    logging.warning ("Ignoring subscript not using 'p'")
            #
            # FIXME check if subscripting any of the known symbols
            #
            if isinstance (node.value, ast.Name):
                name = node.value.id
                if name in self.params.keys ( ):
                    return "dom(%s%s)" % (name, indexing)




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
        self.dim    = None
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
            logging.warning ("ignoring functor parameter '%s'\n" % name)



class StencilFunctor ( ):
    """
    Represents a functor inside a multi-stage stencil.-
    """
    def __init__ (self, node, symbols):
        """
        Constructs a new StencilFunctor:

            node    the FunctionDef AST node (see
                    https://docs.python.org/3.4/library/ast.html) of the
                    Python function from which this functor will be built;
            symbols data fields and constants gathered from other levels
                    of AST analysis.-
        """
        #
        # a name to uniquely identify this functor
        #
        self.name = None
        #
        # a dictionary for the functor parameters (k=name, v=param)
        #
        self.params = dict ( )
        #
        # the body of the functor is inlined from the 'for' loops
        #
        self.body = None
        #
        # the AST node of the Python function representing this functor
        #
        self.set_ast (node)
        self.symbols = symbols


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
                    self.params[par.name] = par

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
                                             self.params,
                                             self.symbols)
                    self.body.generate_code ( )

