# -*- coding: utf-8 -*-
import sys
import ast
import logging

import numpy as np

from gridtools.symbol import SymbolInspector




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
        self.scope       = scope 
        self.encl_scope  = encl_scope
        self.symbol_insp = SymbolInspector (scope)
        try:
            if len (nodes) > 0:
                self.nodes = nodes
        except TypeError:
            logging.warning ("FunctorBody expects a list of AST nodes.")


    def _analyze_assignment (self, lval_node, rval_node):
        """
        Analyze any known symbols appearing as LValue or RValue in an expression.-
        """
        lvalues = self.symbol_insp.search (lval_node)
        rvalues = self.symbol_insp.search (rval_node)

        for lsymbol in lvalues:
            #
            # lvalues are read/write
            #
            lsymbol.read_only = False
            #
            # lvalues depend on rvalues
            #
            for rsymbol in rvalues:
                self.scope.add_dependency (lsymbol,
                                           rsymbol)


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
            sign = '**'
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
            ret_value = "%s = %s" % (self.visit (tgt),          # lvalue
                                     self.visit (node.value))   # rvalue
            self._analyze_assignment (tgt,
                                      node.value)
        return ret_value


    def visit_Attribute (self, node):
        """
        Generates code for attribute references in Python.-
        """
        name = "%s.%s" % (node.value.id,
                          node.attr)
        symbol = None

        #
        # first look for the symbol within this functor's scope
        #
        if name in self.scope:
            symbol = self.scope[name]
            #
            # try to inline the value of the symbol
            #
            if self.scope.is_constant (name):
                return str (symbol.value)
            else:
                #
                # replacing the dot with underscore gives a valid C++ name
                #
                return name.replace ('.', '_')
        #
        # then within the enclosing scope, so to enforce correct scope shadowing
        #
        elif name in self.encl_scope:
            symbol = self.encl_scope[name]
            #
            # try to inline the value of the symbol
            #
            if self.encl_scope.is_constant (name):
                return str (symbol.value)
            else:
                #
                # non-constant symbols in the enclosing scope 
                # become parameters of this functor
                #
                self.scope.add_parameter (name,
                                          symbol.value,
                                          read_only=symbol.read_only)
                #
                # replacing the dot with underscore gives a valid C++ name
                #
                return name.replace ('.', '_')
        else:
            raise RuntimeError ("Unknown symbol '%s'" % attr_name)


    def visit_AugAssign (self, node):
        """
        Generates code for an operation-assignment node, e.g., expr += expr.-
        """
        sign = self._sign_operator (node.op)
        ret_value = "%s %s= %s" % (self.visit (node.target),
                                   sign,
                                   self.visit (node.value))
        self._analyze_assignment (node.target,
                                  node.value)
        return ret_value


    def transpow(self, base, exp):
        logging.debug ("Exponent type %s" % type(exp))
        if (isinstance (exp, ast.UnaryOp)):
            exp = op.operand.n
        if ( not isinstance(exp, float) and not isinstance(exp, int)):
            logging.warning ("This is neither a float nor an int.  type = %s", type(exp))
            exp = eval(exp)
            logging.debug ("After evaluating it, the new type of the expression is %s", type(exp))

        if ( not isinstance(exp, int)):
            if ( isinstance(exp, float)):
                exp = int(exp)  # Convert float to int
                logging.warning ("WARNING!  The evaluated exponent is a floating point.  Currently, only whole integers can be translated.  Truncating to integer.")
            else:
                logging.error ("Can not determine a number for the exponent (type = %s)", type(exp))
                return "NaN"

        if (exp == 0): 
            return "(1)"
        elif (exp == 1): 
            return "("+str(base)+")"
        elif (exp > 1): 
            if ( not isinstance(base, float) and not isinstance(base, int)):
                val = "*("+str(base)+")"
                return "(({0}){1})".format(base, ''.join([val for num in range(exp-1)]))
            else:
                val = "*"+str(base)
                return "({0}{1})".format(base, ''.join([val for num in range(exp-1)]))
        elif (exp < 0): 
            if ( not isinstance(base, float) and not isinstance(base, int)):
                val = "*("+str(base)+")"
                return "(1/(({0}){1}))".format(base, ''.join([val for num in range(abs(exp)-1)]))
            else:
                val = "*"+str(base)
                return "(1/({0}{1}))".format(base, ''.join([val for num in range(abs(exp)-1)]))


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
            elif (isinstance (op, ast.UnaryOp) ):
                operand.append ('%s' % self.visit_UnaryOp(op) )
            else:
                operand.append ('(%s)' % self.visit (op))

        if (sign == '**'):
            return self.transpow(operand[0], operand[1])
        else:
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
                                      read_only=symbol.read_only)
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
                            raise RuntimeError ("Subscript shifting operation '%s' unknown"
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
        else:
            logging.warning ("Slicing operations cannot be translated")
            

    def visit_UnaryOp (self, node):
        """
        Generates code for an unary operator, e.g., -
        """
        sign = self._sign_operator (node.op)
        return "%s%s" % (sign,
                         self.visit (node.operand))




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


    def __hash__ (self):
        return self.name.__hash__ ( )


    def generate_code (self, src):
        """
        Generates the C++ code of this functor:

            src     the Python source from which the C++ is generated;
                    this is used to display user-friendly error messages.-
        """
        self.body.generate_code (src)


    def get_dependency_graph (self):
        return self.scope.depency_graph


    def translate (self):
        """
        Translates this functor to C++, using the gridtools interface, returning
        a string of rendered file.-
        """
        from gridtools import JinjaEnv

        functor_tpl = JinjaEnv.get_template ("functor.h")
        params       = list (self.scope.get_parameters ( ))

        return functor_tpl.render (functor=self,
                                   params=params)
