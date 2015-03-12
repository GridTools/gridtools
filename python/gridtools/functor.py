# -*- coding: utf-8 -*-
import sys
import ast
import logging

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
        self.dim    = None
        self.input  = None
        self.output = None
        self.set_name (name)


    def set_name (self, name):
        """
        Sets a new name to this functor parameter.-
        """
        #
        # do not add 'self' as a functor parameter
        #
        if name != 'self':
            self.name = name
            #
            # temporary parameters are not 'input' nor 'output'
            #                                                                        
            self.input  = self.name.startswith ('in_')
            self.output = self.name.startswith ('out_')
        else:
            self.name = None



class FunctorBody (ast.NodeVisitor):
    """
    Represents the Do( ) function of a stencil's functor in AST form.-
    """
    def __init__ (self, nodes, scope, encl_scope):
        """
        Constructs a functor body object using the received node:

            nodes       an AST-node list representing the body of this functor;
            scope       the symbols scope of this functor;
            encl_scope  the enclosing scope of symbols that are visible to this
                        functor.-
        """
        self.scope      = scope 
        self.encl_scope = encl_scope
        try:
            if len (nodes) > 0:
                self.nodes = nodes
        except TypeError:
            logging.warning ("FunctorBody expects a list of AST nodes.")


    def _sign_operator (self, op):
        """
        Returns the sign representation of an arithmetic operation.-
        """
        if isinstance (op, ast.Add) or isinstance (op, ast.UAdd):
            sign = '+'
        elif isinstance (op, ast.Sub) or isinstance (op, ast.USub):
            sign = '-'
        elif isinstance (op, ast.Mult):
            sign = '*'
        elif isinstance (op, ast.Div):
            sign = '/'
        elif isinstance (op, ast.Pow):
            #
            # TODO: translate power to a multiplication
            #
            sign = None
            logging.warning ("Cannot translate 'x**y'")
        else:
            sign = None
            logging.warning ("Cannot translate '%s'" % str (op))
        return sign

         
    def generate_code (self, src):
        """
        Generates C++ code from the AST backing this object:

            src     the Python source used to display user-friendly 
                    error messages.-
        """
        self.cpp_src = ''
        for n in self.nodes:
            try:
                self.cpp_src += self.visit (n)
                if self.cpp_src:
                    self.cpp_src = "%s;\n\t\t" % self.cpp_src

            except RuntimeError as e:
                #
                # preprocess the source code to correctly display the line,
                # because comments are lost in the AST translation
                #
                # FIXME: comment_offset is not correctly calculated
                #
                src_lines      = src.split ('\n')
                comment_offset = 0
                correct_lineno = n.lineno + comment_offset
                source_line    = src_lines[correct_lineno].strip (' ')
                raise type(e) ("at line %d:\n\t%s" % (correct_lineno,
                                                      source_line))


    def visit_Assign (self, node):
        """
        Generates code from an Assignment node, i.e., expr = expr.-
        """
        for tgt in node.targets:
            return "%s = %s" % (self.visit (tgt),          # lvalue
                                self.visit (node.value))   # rvalue


    def visit_Attribute (self, node):
        """
        Generates code for attribute references in Python.-
        """
        name = "%s.%s" % (node.value.id,
                          node.attr)
        symbol = None

        #
        # looking for the symbol name in this order forces
        # correct symbol name shadowing
        #
        if name in self.scope:
            symbol = self.scope[name]
        elif name in self.encl_scope:
            symbol = self.encl_scope[name]
            #
            # existing symbols in the enclosing scope
            # are parameters to this functor
            #
            self.scope.add_parameter (name,
                                      symbol.value,
                                      read_only=self.encl_scope.is_parameter (name,
                                                                              read_only=True))
        else:
            raise RuntimeError ("Unknown symbol '%s'" % attr_name)
        #
        # do not replace strings or NumPy arrays
        #
        if (isinstance (symbol.value, str) or
            isinstance (symbol.value, np.ndarray)):
            #
            # replacing the dot with underscore gives a valid C++ name
            #
            return name.replace ('.', '_')
        else:
            #
            # otherwise, we just inline the value of the symbol
            #
            return str (symbol.value)


    def visit_AugAssign (self, node):
        """
        Generates code for an operation-assignment node, e.g., expr += expr.-
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


    def visit_Name (self, node):
        """
        Generates code for a variable name, e.g., a functor parameter.-
        """
        name   = node.id
        symbol = None

        #
        # looking for the symbol name in this order forces
        # correct symbol name shadowing
        #
        if name in self.scope:
            symbol = self.scope[name]
        elif name in self.encl_scope:
            symbol = self.encl_scope[name]
            #
            # existing symbols in the enclosing scope
            # are parameters to this functor
            #
            self.scope.add_parameter (name,
                                      symbol.value,
                                      read_only=self.encl_scope.is_parameter (name,
                                                                              read_only=True))
        else:
            raise RuntimeError ("Unknown symbol '%s'" % name)
        #
        # try to inline the value of this symbol
        #
        if (isinstance (symbol.value, str) or
            isinstance (symbol.value, np.ndarray)):
            return name 
        else:
            return str (symbol.value)


    def visit_Num (self, node):
        """
        Returns the number in this node.-
        """
        return str (node.n)


    def visit_Subscript (self, node):
        """
        Generates code from Subscript node, i.e., expr[expr].-
        """
        if isinstance (node.slice, ast.Index):
            #
            # subscript with shifting?
            #
            if isinstance (node.slice.value, ast.BinOp):
                if isinstance (node.slice.value.op, ast.Add):
                    indexing = '('
                    for e in node.slice.value.right.elts:
                        if isinstance (e, ast.Num):
                            indexing += "%s," % str (e.n)
                        #
                        # shifting with negative numbers
                        #
                        elif isinstance (e, ast.UnaryOp):
                            indexing += "%s%s," % (self._sign_operator (e.op),
                                                   str (e.operand.n))
                        else:
                            raise RuntimeError ("Subscript shifting operation %s unknown"
                                                % str (e.op))
                    #
                    # strip the last comma off
                    #
                    indexing = '%s)' % indexing[:-1]

                    #
                    # range detection for data fields
                    #
                    if isinstance (node.value, ast.Name):
                        name   = self.visit_Name (node.value)
                        symbol = self.scope[name]
                        #
                        # range only makes sense for data fields, i.e., NumPy arrays
                        #
                        if isinstance (symbol.value, np.ndarray):
                            symbol.set_range (eval (indexing))
                else:
                    indexing = ''
                    logging.warning ("Subscript shifting only supported with '+'")
            #
            # subscript without shifting
            #
            elif isinstance (node.slice.value, ast.Name):
                if node.slice.value.id == 'p':
                    indexing = '( )'
                else:
                    #
                    # FIXME should previously extract the subscripting symbol
                    #       over 'get_interior_points'
                    #
                    indexing = ''
                    logging.warning ("Ignoring subscript not using 'p'")

        return "eval(%s%s)" % (self.visit (node.value), 
                               indexing)
            



class Functor ( ):
    """
    Represents a functor inside a multi-stage stencil.-
    """
    def __init__ (self, name, node, scope, encl_scope):
        """
        Constructs a new StencilFunctor:

            name        a name to uniquely identify this functor;
            node        the For AST node (see
                        https://docs.python.org/3.4/library/ast.html) of the
                        Python comprehention from which this functor is
                        constructed;
            scope       the symbols scope of this functor;
            encl_scope  the enclosing scope of symbols that are visible to this
                        functor.-
        """
        self.name       = name
        self.scope      = scope
        self.encl_scope = encl_scope
        #
        # the root AST node of the for-loop representing this functor
        #
        if isinstance (node, ast.For):
            self.node = node 
        else:
            raise TypeError ("Functor's root AST node should be 'ast.For'")
        #
        # the body of this functor
        #
        self.body = FunctorBody (self.node.body,
                                 self.scope,
                                 self.encl_scope)


    def generate_code (self, src):
        """
        Generates the C++ code of this functor:

            src     the Python source from which the C++ is generated;
                    this is used to display user-friendly error messages.-
        """
        self.body.generate_code (src)

