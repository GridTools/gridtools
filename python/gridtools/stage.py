# -*- coding: utf-8 -*-
import sys
import ast
import logging

import numpy as np

from gridtools.symbol import Scope, SymbolInspector




class StageBody (ast.NodeVisitor):
    """
    Represents the Do( ) function of a stencil's functor in AST form.-
    """
    symbol_inspector = SymbolInspector ( )

    def __init__ (self, nodes, scope, stencil_scope):
        """
        Constructs a functor body object
        :param nodes:         an AST-node list representing the body of this 
                              functor
        :param scope:         the symbols scope of this functor
        :param stencil_scope: the enclosing scope of symbols that are visible 
                              to this stage
        :raise TypeError:     if nodes is not iterable
        """
        self.scope         = scope 
        self.stencil_scope = stencil_scope
        try:
            if len (nodes) > 0:
                self.nodes = nodes
        except TypeError:
            raise TypeError ("StageBody expects a list of AST nodes.")


    def _analyze_assignment (self, lval_node, rval_node):
        """
        Analyze any known symbols appearing as LValue or RValue
        :param lval_node: AST node of the expression appearing as LValue
        :param rval_node: AST node of the expression appearing as RValue
        :return:
        """
        lvalues = StageBody.symbol_inspector.search (lval_node, 
                                                     self.scope)
        rvalues = StageBody.symbol_inspector.search (rval_node, 
                                                     self.scope)
        for lsymbol in lvalues:
            #
            # lvalues (and their aliases) are read/write
            #
            if self.scope.is_alias (lsymbol.name):
                self.scope[lsymbol.value].read_only = False
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
        elif isinstance (op, ast.Not):
            sign = '!'
        else:
            sign = None
            logging.warning ("Cannot translate '%s'" % str (op))
        return sign

         
    def _transpow (self, base, exp):
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
                logging.warning ("The evaluated exponent is a floating point.  Currently, only whole integers can be translated.  Truncating to integer.")
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


    def generate_code (self):
        """
        Generates C++ code from the AST backing this object
        :return:
        """
        self.cpp_src = ''
        for n in self.nodes:
            try:
                self.cpp_src += self.visit (n)
                if self.cpp_src:
                    self.cpp_src = "%s;\n\t\t" % self.cpp_src
            except RuntimeError as e:
                #
                # TODO: preprocess the Python source code to correctly display 
                # the line where the error occurred, because comments are lost 
                # in the AST translation
                #
                #src_lines      = src.split ('\n')
                #comment_offset = 0
                #correct_lineno = n.lineno + comment_offset
                #source_line    = src_lines[correct_lineno].strip (' ')
                raise type(e)

    def visit_CompOp(self, node):
        op = "None"
        if (isinstance (node, ast.Eq)):
            op = "=="
        if (isinstance (node, ast.NotEq)):
            op = "!="
        if (isinstance (node, ast.Lt)):
            op = "<"
        if (isinstance (node, ast.LtE)):
            op = "<="
        if (isinstance (node, ast.Gt)):
            op = ">"
        if (isinstance (node, ast.GtE)):
            op = ">="
        if (isinstance (node, ast.Is)):
            op = "None"
            raise NotImplementedError ("Translation of Python 'Is' is not currently supported.")
        if (isinstance (node, ast.IsNot)):
            op = "None"
            raise NotImplementedError ("Translation of Python 'IsNot' is not currently supported.")
        if (isinstance (node, ast.In)):
            op = "None"
            raise NotImplementedError ("Translation of Python 'In' is not currently supported.")
        if (isinstance (node, ast.NotIn)):
            op = "None"
            raise NotImplementedError ("Translation of Python 'NotIn' is not currently supported.")
        return op

    def visit_Compare(self, node):
        ret_value = "%s" % self.visit(node.left)
        for cmpop in node.ops:
            ret_value = "%s %s" % (ret_value, self.visit_CompOp(cmpop))

        for cmp in node.comparators:
            ret_value = "%s %s" % (ret_value, self.visit(cmp))

        return ret_value

    def _boolean_operator (self, op):
        """
        Returns the sign representation of an arithmetic operation.-
        """
        if isinstance (op, ast.And):
            sign = '&&'
        elif isinstance (op, ast.Or):
            sign = '||'
        else:
            sign = None
            raise RuntimeError("Cannot translate '%s'" % str (op))
        return sign

    def visit_BoolOp(self, node):
        ret_value = ""
        op = self._boolean_operator(node.op)
        nels = len(node.values)
        if(nels > 0):
            ret_value = "(%s" % self.visit(node.values[0])
            for expr in node.values[1:]:
                ret_value = "%s %s %s" % (ret_value, op, self.visit(expr))
            ret_value = "%s)" % ret_value

        return ret_value


    def visit_If (self, node):
        ret_value = "if (%s) {" % self.visit(node.test)
        for stmt in node.body:
            ret_value = "%s %s;" % (ret_value,self.visit(stmt))
        if (node.orelse is not None and len(node.orelse) >= 1):
            ret_value = "%s } else { " % ret_value
            for stmt in node.orelse:
                ret_value = "%s %s;" % (ret_value,self.visit(stmt))

        ret_value = "%s }" % ret_value
        return ret_value

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
                return name
        #
        # then within the enclosing scope, so to enforce correct scope shadowing
        #
        elif name in self.stencil_scope:
            symbol = self.stencil_scope[name]
            #
            # try to inline the value of the symbol
            #
            if self.stencil_scope.is_constant (name):
                return str (symbol.value)
            else:
                #
                # non-constant symbols in the enclosing scope 
                # become parameters of this functor
                #
                self.scope.add_parameter (name,
                                          symbol.value,
                                          read_only=symbol.read_only)
                return name
        else:
            raise RuntimeError ("Unknown symbol '%s'" % name)


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
            return self._transpow(operand[0], operand[1])
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
        elif name in self.stencil_scope:
            symbol = self.stencil_scope[name]
            #
            # existing symbols in the enclosing scope
            # are parameters to this functor
            #
            self.scope.add_parameter (name,
                                      symbol.value,
                                      read_only=symbol.read_only)
        else:
            raise NameError ("Unkown symbol '%s' in functor" % name)
        #
        # resolve aliases before trying to inline
        #
        if self.scope.is_alias (name):
            if symbol.value in self.scope:
                aliased = self.scope[symbol.value]
            elif symbol.value in self.stencil_scope:
                aliased = self.stencil_scope[symbol.value]
            else:
                raise NameError ("Could not dereference alias '%s'" % name)
            self.scope.add_parameter (aliased.name,
                                      aliased.value,
                                      read_only=symbol.read_only)
            return aliased.name
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
                    # access-pattern detection for data fields ...
                    #
                    if (isinstance (node.value, ast.Name) or 
                        isinstance (node.value, ast.Attribute)):
                        name   = self.visit (node.value)
                        symbol = self.scope[name]
                        #
                        # ... only makes sense for NumPy arrays
                        #
                        if isinstance (symbol.value, np.ndarray):
                            symbol.set_access_pattern (eval (indexing))
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

            return "eval(%s%s)" % (self.visit (node.value).replace ('.', '_'), 
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



class StageScope (Scope):
    """
    Stage symbols are organized into scopes that represent code visibility
    blocks.-
    """
    def get_ghost_cell (self):
        """
        Returns the ghost-cell pattern of this stage alone
        :return: a 4-element list describing the ghost cell
        """
        ghost = [0,0,0,0]
        for sym in self.get_all ( ):
            if sym.access_pattern is not None:
                for idx in range (len (sym.access_pattern)):
                    ghost[idx] += sym.access_pattern[idx]
        return ghost



class Stage ( ):
    """
    Represents a stage inside a stencil.-
    """
    def __init__ (self, name, node, stencil_scope):
        """
        Constructs a new StencilStage
        :param name:          a name to uniquely identify this functor
        :param node:          the For AST node of the comprehention from which
                              this functor is constructed
        :param stencil_scope: the scope of symbols at stencil level
        :raise TypeError:     if the passed node is of the incorrect type
        :return:
        """
        self.name          = name
        self.scope         = StageScope ( )
        self.stencil_scope = stencil_scope
        #
        # the ghost-cell access pattern of this stage
        #
        self.ghost_cell    = None
        #
        # whether this stage is executed independently from other stages
        #
        self._independent  = False
        #
        # the root AST node of the for-loop representing this functor
        #
        if isinstance (node, ast.For):
            self.node = node 
        else:
            raise TypeError ("Stage's root AST node should be 'ast.For'")
        #
        # the body of this functor
        #
        self.body = StageBody (self.node.body,
                               self.scope,
                               self.stencil_scope)


    def __hash__ (self):
        return self.name.__hash__ ( )


    def __repr__ (self):
        return self.name


    def generate_code (self):
        """
        Generates the C++ code of this functor
        :return:
        """
        self.body.generate_code ( )


    def get_data_dependency (self):
        return self.scope.data_dependency


    @property
    def independent (self):
        return self._independent


    @independent.setter
    def independent (self, value):
        self._independent = bool (value)
        #
        # have to rebuild the stage-execution graph
        #
        self.stencil_scope.build_execution_path ( )


    def translate (self):
        """
        Translates this functor to C++, using the gridtools interface, returning
        a string of rendered file.-
        """
        from gridtools import JinjaEnv

        functor_tpl = JinjaEnv.get_template ("functor.h")
        params      = list (self.scope.get_parameters ( ))

        return functor_tpl.render (functor=self,
                                   params=params)

