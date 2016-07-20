# -*- coding: utf-8 -*-
import ast
import logging

import numpy as np

from gridtools.symbol import Scope, SymbolInspector




class StageBody (ast.NodeVisitor):
    """
    Represents the Do( ) function of a stencil's stage in AST form.-
    """
    symbol_inspector = SymbolInspector ( )

    def __init__ (self, stage_name, nodes, scope, stencil_scope):
        """
        Constructs a stage body object
        :param stage_name:    name of the stage this body belongs to
        :param nodes:         an AST-node list representing the body of this
                              stage
        :param scope:         the symbols scope of this stage
        :param stencil_scope: the enclosing scope of symbols that are visible
                              to this stage
        :raise TypeError:     if nodes is not iterable
        """
        self.stage_name    = stage_name
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
            #
            # When assigning to a local variable, if rvalue is a scalar, assign
            # its numerical value to the local symbol, otherwise set the
            # symbol value as None.
            # Setting the value to None won't break things because a local
            # symbol value is inlined only if it is a number.
            #
            if self.scope.is_local (lsymbol.name):
                if isinstance (rval_node, ast.Num):
                    self.scope.add_local (lsymbol.name, rval_node.n)
                else:
                    self.scope.add_local (lsymbol.name)


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
            lvalue = self.visit (tgt)
            rvalue = self.visit (node.value)
            #
            # Declare type for first assignment of local variable
            #
            if self.scope.is_local (lvalue) and self.scope[lvalue].value is None:
                lvalue_type = "double"
                lvalue  =  lvalue_type + " " + lvalue
            #
            # Create assignment string
            #
            ret_value = "%s = %s" % (lvalue,
                                     rvalue)
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
        # first look for the symbol within this stage's scope
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
                # become parameters of this stage
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
        Generates code for a variable name, e.g., a stage parameter.-
        """
        name   = node.id
        symbol = None

        #
        # first look for the symbol within this stage's scope
        #
        if name in self.scope:
            symbol = self.scope[name]
            #
            # resolve aliases before trying to inline
            #
            if self.scope.is_alias (name):
                if symbol.value in self.stencil_scope:
                    aliased = self.stencil_scope[symbol.value]
                else:
                    raise NameError ("Could not dereference alias '%s'" % name)
                self.scope.add_parameter (aliased.name,
                                          aliased.value,
                                          read_only=symbol.read_only)
                return aliased.name
            #
            # If symbol is a local variable, inline its value only if it is a
            # scalar integer or floating point number in a Load context
            #
            if self.scope.is_local (name):
               if (isinstance (symbol.value, int)
                   or isinstance (symbol.value, float )):
                       if isinstance (node.ctx, ast.Load):
                           return symbol.value
                       else:
                           return name
               else:
                   return name
        #
        # then within the enclosing scope, so to enforce correct scope shadowing
        #
        elif name in self.stencil_scope:
            symbol = self.stencil_scope[name]
            #
            # existing symbols in the enclosing scope
            # are parameters to this stage
            #
            self.scope.add_parameter (name,
                                      symbol.value,
                                      read_only=symbol.read_only)
        #
        # Name is not in any known scope: check context to see if we are storing
        # a new local variable
        #
        elif isinstance (node.ctx, ast.Store):
            #
            # Value will be resolved by the function visiting the assignment
            #
            self.scope.add_local (name)
            return name
        else:
            raise NameError ("Unknown symbol '%s' in stage '%s'"
                             % (name, self.stage_name))
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
    def __init__ (self):
        super ( ).__init__ ( )
        #
        # The ghost-cell pattern of this stage
        #
        self.ghost_cell = None



class VerticalRegion ( ):
    """
    Represents a vertical region (an interval in the k direction) on which a
    Stage can operate
    """
    def __init__ (self, name, array_name, slice_start_node=None, slice_end_node=None):
        """
        Constructs a new VerticalRegion
        :param name:                name of this vertical region
        :param array_name:          a string representing the name of the array
                                    being sliced
        :param slice_start_node:    an AST Slice node containing the starting
                                    value of the region
        :param slice_end_node:      an AST Slice node containing the starting
                                    value of the region
        """
        self.name             = name
        self.array_name       = array_name
        self.slice_start_node = slice_start_node
        self.slice_end_node   = slice_end_node
        #
        # Cell indexes marking the beginning and the end of the vertical region
        #
        self.start_k = None
        self.end_k   = None
        #
        # Indexes of the splitters (from the dict mantained by the stencil)
        # for this vertical region.
        #
        self.start_splitter = None
        self.end_splitter   = None


    def set_splitters (self, splitters):
        """
        Sets the splitter values for this region as defined in the GridTools
        backend

        :param splitters:   A dictionary containing splitter values for the
                            stencil. The dict keys are the cell indexes in the k
                            direction where the splitters are located. The dict
                            values are the ordinal indices of the splitters.
        """
        self.start_splitter = splitters[self.start_k]
        self.end_splitter = splitters[self.end_k]


    def find_slice_indexes (self, scope, stencil_scope):
        """
        Find slice indexes from the information
        contained in the slice nodes and array_name

        :param scope:         The scope of the stage this region belongs to
        :param stencil_scope: The scope of the stencil this stage belongs to
        """
        #
        # retrieve the symbol of the sliced array, resolving alias if necessary
        #
        try:
            array_sym = scope[self.array_name]
            if scope.is_alias (array_sym):
                #
                # If the stage is defined through a function, the array symbol value
                # will be an alias to a stencil scope symbol; we have to resolve the
                # alias to get the actual array
                # Otherwise, if the stage is defined through a for loop in the kernel,
                # array_sym's value already contains the numerical array, and no
                # lookup in the stencil scope symbol has to be performed.
                # Remember that the stage scope also contains the stencil-level
                # symbols that correspond to aliases! This means that we don't need
                # the stencil scope object to get to the underlying numerical array.
                #
                array_sym = scope[array_sym.value]
        except KeyError:
            raise KeyError ("Error while recovering array symbol %s for vertical region %s. Check the call to get_interior_points()."
                            % (self.array_name, self.name))
        #
        # set indexes based on the given slicing limits
        #
        if self.slice_start_node is None:
            #
            # set initial index if no slicing limit was given
            #
            start_idx = 0
        elif isinstance (self.slice_start_node, ast.Num):
            start_idx = int (self.slice_start_node.n)
        else:
            raise NotImplementedError ("Only constants are accepted when slicing fields")

        if self.slice_end_node is None:
            #
            # set final index if no slicing limit was given
            #
            end_idx = array_sym.value.shape[2]
        elif isinstance (self.slice_end_node, ast.Num):
            end_idx = int (self.slice_end_node.n)
        else:
            raise NotImplementedError ("Only constants are accepted when slicing fields")
        #
        # check the indexes are within the field bounds and in correct order
        #
        if (start_idx > array_sym.value.shape[2] or
            end_idx   > array_sym.value.shape[2]):
            raise ValueError ("Slicing for field '%s' is out of bounds" % array_sym)

        if start_idx < 0 or end_idx < 0:
            raise ValueError ("Cannot use a negative index in vertical region '%s'"
                              % self.name)
        else:
            self.start_k = start_idx

        if end_idx < start_idx:
            raise ValueError (("End index in vertical region '%s'" % self.name)
                              + "is smaller than its start counterpart" )
        else:
            self.end_k = end_idx

        return (self.start_k, self.end_k)



class Stage ( ):
    """
    Represents a stage inside a stencil.-
    """
    def __init__ (self, name, node, stencil_scope):
        """
        Constructs a new StencilStage

        :param name:          a name to uniquely identify this stage
        :param node:          the For AST node of the comprehention from which
                              this stage is constructed
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
        # Input and output data fields for this stage
        #
        self.inputs        = None
        self.outputs       = None
        #
        # whether this stage could be executed in parallel with other stages
        # inside the stencil (see HorizontalDiffusion test for fluxes I and J)
        #
        self._independent  = False
        #
        # The list of vertical regions for this stage
        #
        self.vertical_regions = []
        #
        # the root AST node of the for-loop representing this stage
        #
        if isinstance (node, ast.For):
            self.node = node
        else:
            raise TypeError ("Stage's root AST node should be 'ast.For'")
        #
        # the body of this stage
        #
        self.body = StageBody (self.name,
                               self.node.body,
                               self.scope,
                               self.stencil_scope)


    def __hash__ (self):
        return self.name.__hash__ ( )


    def __repr__ (self):
        return self.name


    def add_vertical_region (self, array_name, slice_start_node, slice_end_node):
        """
        Adds a vertical region to this stage
        :param slice_start_node:    an AST Slice node containing the starting
                                    value of the region
        :param slice_end_node:      an AST Slice node containing the starting
                                    value of the region
        :param array_name:          a string representing the name of the array
                                    being sliced
        :return:
        """
        vr_name = '%s_VR_%03d' % (self.name, len (self.vertical_regions))
        region = VerticalRegion (vr_name,
                                 array_name,
                                 slice_start_node,
                                 slice_end_node)
        self.vertical_regions.append (region)
        logging.debug ("Vertical Region '%s' created" % vr_name)


    def generate_code (self):
        """
        Generates the C++ code of this stage

        :return:
        """
        self.body.generate_code ( )


    def find_slice_indexes (self, stencil_scope):
        """
        Find slice indexes for each vertical region of this stage

        :param stencil_scope: The symbol scope of the stencil this stage
                              belongs to
        :return:
        """
        stg_slice_indexes = list ( )
        for vr in self.vertical_regions:
            vr_slice = vr.find_slice_indexes (self.scope,
                                              stencil_scope)
            stg_slice_indexes.append (vr_slice)
        #
        # make sure the vertical regions do not overlap
        #
        stg_slice_indexes = sorted (stg_slice_indexes)
        for i in range (len (stg_slice_indexes) - 1):
            (start_k, end_k) = stg_slice_indexes[i]
            if end_k >= stg_slice_indexes[i + 1][0]:
                raise ValueError ("Vertical regions overlap within stage '%s'"
                                  % self.name)
        #
        # save the splitters for each vertical region
        #
        assert (len (self.vertical_regions) == len (stg_slice_indexes))

        return stg_slice_indexes


    def get_data_dependency (self):
        """
        Return the data dependency graph for this stages's scope
        """
        return self.scope.data_dependency


    def identify_IO_fields (self):
        """
        Tries to identify input and output data fields for this stage
        :return:
        """
        #
        # Look for IO using stage's data dependencies
        #
        logging.debug('Probing IO for Stage: %s' % self.name)
        self.outputs = []
        self.inputs = []
        data_dep = self.get_data_dependency()
        for node in data_dep.nodes_iter():
            #
            # Output data have no predecessors
            #
            if not data_dep.predecessors(node.name) and not self.scope.is_local(node):
                self.outputs.append(node)
            #
            # Input nodes have no successors
            #
            if not data_dep.successors(node.name) and not self.scope.is_local(node):
                self.inputs.append(node)
        #
        # Non-local self-looping nodes are not allowed
        # TODO: Only allow self-assignment if access extent is [0,0], complying
        # with Gridtools' data dependency rules. For more information, see
        # https://github.com/eth-cscs/gridtools/wiki/Data-Dependencies-Analysis-in-GridTools
        #
        for node in data_dep.nodes_with_selfloops():
#            self.inputs.append(node)
#            self.outputs.append(node)
            raise ValueError ("Assigning a non-local data field to itself is not allowed.")
        logging.debug('\tStage scope Input data: %s' % self.inputs)
        logging.debug('\tStage scope Output data: %s' % self.outputs)
        #
        # Resolve aliases at stencil scope, substituting the alias with the
        # corresponding symbol, that can be found inside the symbol table!
        #
        for i, data in enumerate(self.inputs):
            if data.kind == 'alias':
                self.inputs[i] = self.scope.symbol_table[data.value]
        for i, data in enumerate(self.outputs):
            if data.kind == 'alias':
                self.outputs[i] = self.scope.symbol_table[data.value]
        logging.debug('\tStencil scope Input data: %s' % self.inputs)
        logging.debug('\tStencil scope Output data: %s' % self.outputs)


    @property
    def independent (self):
        return self._independent


    @independent.setter
    def independent (self, value):
        self._independent = bool (value)


    def set_splitters (self, splitters):
        """
        Sets the splitters values for this stage's vertical regions

        :param splitters:   A dictionary containing splitter values for the
                            stencil. The dict keys are the cell indexes in the k
                            direction where the splitters are located. The dict
                            values are the ordinal indices of the splitters.
        """
        for vr in self.vertical_regions:
            vr.set_splitters (splitters)


    def translate (self):
        """
        Translates this stage to C++, using the gridtools interface, returning
        a string of rendered file.-
        """
        from gridtools import JinjaEnv

        stage_tpl = JinjaEnv.get_template ("stage.h")
        params      = list (self.scope.get_parameters ( ))

        return stage_tpl.render (stage=self,
                                 params=params)
