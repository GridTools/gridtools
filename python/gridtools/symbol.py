# -*- coding: utf-8 -*-
import ast
import logging
import numpy    as np
import networkx as nx




class Symbol (object):
    """
    When traversing the AST of a user-generated stencil, the recognized
    symbols are instances of this class.-
    """
    KINDS = (
        #
        # the 'param' kind indicates a stencil and/or stage data-field parameter
        #
        'param',
        #
        # the 'temp' kind indicates a temporary data field
        #
        'temp',
        #
        # the 'alias' kind indicates a variable or parameter which holds a
        # different name for another known symbol
        #
        'alias',
        #
        # the 'const' kind indicates a variable that may POTENTIALLY be
        # replaced by a value. This may happen by inlining them after using
        # runtime resolution
        #
        'const')


    def __init__ (self, name, kind, value=None):
        """
        Creates a new symbol instance:

            name    the name of the symbol;
            kind    a valid Symbol.KINDS value;
            value   the value of the symbol.-
        """
        assert (kind in Symbol.KINDS)

        self.name  = name
        self.kind  = kind
        self.value = value

        #
        # an access pattern defines the extent of data accessed during the
        # execution of a stencil stage; its structure is defined as follows:
        #
        #   (minimum negative index accessed in _i_,
        #    maximum positive index accessed in _i_,
        #    minimum negative index accessed in _j_,
        #    maximum positive index accessed in _j_)
        #
        self.access_pattern = None

        #
        # a flag indicating whether this symbol is read-only
        #
        self.read_only      = True


    def __eq__ (self, other):
        return self.__hash__ ( ) == other.__hash__ ( )

    def __ne__ (self, other):
        return not self.__eq__ (other)

    def __hash__ (self):
        return self.name.__hash__ ( )

    def __repr__ (self):
        return self.name


    def set_access_pattern (self, offset):
        """
        Sets or updates the access pattern of this symbol
        :param offset: the pattern is defined as (access offset in 'i',
                                                  access offset in 'j',
                                                  access offset in 'k')
        :raise IndexError: if the access pattern
        :return:
        """
        try:
            i, j, k = offset

            if self.access_pattern is None:
                self.access_pattern = [0, 0, 0, 0]

            if self.access_pattern[0] > i:
                self.access_pattern[0] = i
            elif self.access_pattern[1] < i:
                self.access_pattern[1] = i

            if self.access_pattern[2] > j:
                self.access_pattern[2] = j
            elif self.access_pattern[3] < j:
                self.access_pattern[3] = j

        except IndexError:
            logging.error ("Access-offset descriptor %s is invalid" % offset)



class SymbolInspector (ast.NodeVisitor):
    """
    Inspects the AST looking for known symbols.-
    """
    def __init__ (self):
        self.scope         = None
        self.symbols_found = None


    def search (self, node, scope):
        """
        Returns a list of symbols belonging to the current scope that are
        found in the AST, the root of which is 'node'
        :param node:  the AST node from where to start the search
        :param scope: the scope of symbols
        :return:      a set containing all the symbols found
        """
        self.scope         = scope
        self.symbols_found = set ( )
        self.visit (node)
        return self.symbols_found


    def visit_Attribute (self, node):
        """
        Looks for this attribute in the current scope.-
        """
        name = "%s.%s" % (node.value.id,
                          node.attr)
        if name in self.scope:
            self.symbols_found.add (self.scope[name])


    def visit_Name (self, node):
        """
        Looks for this name in the current scope.-
        """
        name = node.id
        if name in self.scope:
            self.symbols_found.add (self.scope[name])


    def visit_Subscript (self, node):
        self.visit (node.value)



class Scope (object):
    """
    Defines the scope within which a symbol is defined.-
    """
    def __init__ (self):
        #
        # the symbol table is kept as a dict, i.e.:
        #
        #       A = (k=symbol name, v=symbol object)
        #
        self.symbol_table = dict ( )
        #
        # a data-dependency graph
        #
        #
        self.data_dependency = nx.DiGraph ( )


    def __contains__ (self, name):
        """
        Returns True if a symbol with 'name' is found in this scope.-
        """
        return name in self.symbol_table.keys ( )


    def __getitem__ (self, name):
        """
        Returns 'name' symbol from this scope.-
        """
        return self.symbol_table[name]


    def add_alias (self, name, known_symbol_name):
        """
        Adds a alias to another known symbol:

            name                name of this symbol alias;
            known_symbol_name   name of the known symbol this alias refers to.-
        """
        if known_symbol_name is not None:
            self.add_symbol (Symbol (name,
                                     'alias',
                                     known_symbol_name))
            logging.debug ("Alias '%s' refers to '%s'" % (name,
                                                          known_symbol_name))
        else:
            raise ValueError ("Aliases should point to known symbols")


    def add_constant (self, name, value=None):
        """
        Adds a constant this scope:

            name    name of this constant;
            value   its value.-
        """
        if value:
            try:
                value = float (value)
                logging.debug ("Constant '%s' has value %.3f" % (name,
                                                                value))
            except TypeError:
                if isinstance (value, np.ndarray):
                    logging.debug ("Constant '%s' is a NumPy array %s" % (name,
                                                                         value.shape))
                else:
                    logging.debug ("Constant '%s' has value %s" % (name,
                                                                  value))
        else:
            logging.debug ("Constant '%s' is None" % name)
        self.add_symbol (Symbol (name, 'const', value))


    def add_dependency (self, left_symbol, right_symbol):
        """
        Registers a data dependency between two symbols
        :param left_symbol:  symbol appearing as LValue
        :param right_symbol: symbol appearing as RValue
        :raise ValueError:   if trying to add a dependency between non-existent
                             symbols
        :return:
        """
        if (left_symbol.name in self) and (right_symbol.name in self):
            self.data_dependency.add_edge (left_symbol,
                                           right_symbol)
        else:
            raise ValueError ("Trying to add a data dependency between non-existent symbols")


    def add_dependencies (self, deps):
        """
        Adds a sequence of data dependencies, each of which must be given
        as a 2-tuples of Symbols (u,v)
        :param deps:       an iterable of Symbol pairs (s1, s2)
        :raise ValueError: if trying to add a dependency between non-existent
                           symbols
        :return:
        """
        for d in deps:
            try:
                self.add_dependency (d[0], d[1])
            except ValueError:
                #
                # aliases should be dereferenced
                #
                new_dep = [None, None]
                for i in range (2):
                    if isinstance (d[i].value, str) and d[i].value in self:
                        new_dep[i] = self[d[i].value]
                    else:
                        new_dep[i] = d[i]
                self.add_dependency (new_dep[0], new_dep[1])


    def add_parameter (self, name, value=None, read_only=True):
        """
        Adds a parameter data field to this scope:

            name        the name of the parameter;
            value       its value;
            read_only   whether the parameter is read-only (default).-
        """
        if name not in self:
            self.add_symbol (Symbol (name, 'param', value))
        self[name].kind      = 'param'
        self[name].read_only = read_only
        if value is not None:
            self[name].value = value
        if value is not None:
            if isinstance (value, np.ndarray):
                logging.debug ("Parameter '%s' has dimension %s" % (name,
                                                                   value.shape))
            else:
                try:
                    logging.debug ("Parameter '%s' has value %.3f" % (name,
                                                                     float (value)))
                except ValueError:
                    logging.debug ("Parameter '%s' has value %s" % (name,
                                                                   value))


    def add_temporary (self, name, value=None):
        """
        Adds a temporary data field to this scope
        :param name: the name of the temporary data field
        :param value: the value of the temporary data field
        :raise TypeError: if the given value is not a NumPy array
        :raise ValueError: if the given value is None
        :return:
        """
        if value is not None:
            if isinstance (value, np.ndarray):
                #
                # add the field as a temporary
                #
                self.add_symbol (Symbol (name, 'temp', value))
            else:
                raise TypeError ("Temporary data field '%s' should be a NumPy array not '%s'" %
                                 (name, type (value)))
        else:
            raise ValueError ("Temporary data field '%s' cannot be None" % name)


    def add_symbol (self, symbol):
        """
        Adds or updated the received symbol in this scope
        :param symbol: the Symbol object to add or update
        :return:
        """
        self.symbol_table[symbol.name] = symbol


    def dump (self):
        """
        Prints a formatted list of all in this scope.-
        """
        for k,v in self.symbol_table.items ( ):
            try:
                #
                # for NumPy arrays we display only its dimensions
                #
                val = v.value.shape
            except AttributeError:
                val = str (v.value)

            logging.debug ("\t[%s]\t(%s)\t%s: %s" % (k,
                                                     v.kind,
                                                     v.name,
                                                     val))


    def is_alias (self, name):
        """
        Returns True is symbol with 'name' is an alias.-
        """
        return name in [a.name for a in self.get_all ( ) if a.kind == 'alias']


    def is_constant (self, name):
        """
        Returns True if symbol with 'name' is a constant.-
        """
        return name in [t.name for t in self.get_constants ( )]


    def is_parameter (self, name):
        """
        Returns True is symbol 'name' is a parameter in this scope.-
        """
        return name in [p.name for p in self.get_parameters ( )]


    def is_temporary (self, name):
        """
        Returns True if symbol 'name' is a temporary data field.-
        """
        return name in [t.name for t in self.get_temporaries ( )]


    def get_all (self, kinds=None):
        """
        Returns all symbols in this scope sorted by name:

            kinds   returns only symbol kinds contained in this list.-
        """
        sorted_names = sorted (self.symbol_table.keys ( ))
        if kinds is None:
            for n in sorted_names:
                yield self.symbol_table[n]
        else:
            for n in sorted_names:
                if self.symbol_table[n].kind in kinds:
                    yield self.symbol_table[n]


    def get_constants (self):
        """
        Returns a sorted list of all constants in this scope.-
        """
        return self.get_all (['const'])


    def get_parameters (self):
        """
        Returns a sorted list of all parameters in this scope.-
        """
        return self.get_all (['param'])


    def get_temporaries (self):
        """
        Returns a sorted list of all temporary data fields at this scope.-
        """
        return self.get_all (['temp'])


    def remove (self, name):
        """
        Removes symbol with 'name'.-
        """
        self.symbol_table.pop (name)




class StencilScope (Scope):
    """
    Stencil symbols are organized into scopes that alter the visibility
    of the variables defined in the stencil or any of its stages.
    Basically, if a symbol is defined at stencil level, the stencil itself and
    all its stages can access it. If it is defined at stage level, it is
    accesible only within that stage.-
    """
    def __init__ (self):
        super ( ).__init__ ( )
        #
        # a graph describing the execution path of the stages within the stencil
        #
        self.stage_execution = nx.DiGraph ( )
        #
        # the minimal required halo to correctly execute the stencil
        #
        self.minimum_halo    = None
        #
        # the stencil's source code
        #
        self.py_src          = None


    def _resolve_params (self, stencil, **kwargs):
        """
        Attempts to aquire more information about the discovered parameters
        using runtime information from the user's stencil
        :param stencil: the user's stencil instance
        :return:
        """
        for k,v in kwargs.items ( ):
            if self.is_parameter (k):
                if isinstance (v, np.ndarray):
                    #
                    # update the value of this parameter
                    #
                    self.add_parameter (k,
                                        v,
                                        read_only=self[k].read_only)
                    #
                    #
                    # check the dimensions of different parameters match
                    #
                    if stencil.domain is None:
                        stencil.domain = v.shape
                    elif stencil.domain != v.shape:
                        logging.warning ("Dimensions of parameter '%s':%s do not match %s" %
                                        (k, v.shape, stencil.domain))
                else:
                    logging.warning ("Parameter '%s' is not a NumPy array" % k)


    def add_stage (self, node, prefix='', suffix=''):
        """
        Adds a Stage object to this stencil's scope
        :param node:   the For AST node of the comprehention from which
                       the stage is constructed
        :param prefix: prefix to add to the stage's name
        :param suffix: suffix to add to the stage's name
        :return:       the corresponding Stage object
        """
        from gridtools.stage import Stage

        stage_name = '%s_%s_%03d' % (prefix,
                                     suffix,
                                     len (self.stage_execution))
        stage_obj  = Stage (stage_name,
                            node,
                            self)
        if stage_obj not in self.stage_execution:
            #
            # update the stage execution path
            #
            if len (self.stage_execution) == 0:
                self.stage_execution.add_node (stage_obj)
            else:
                postorder = nx.topological_sort (self.stage_execution,
                                                 reverse=True)
                self.stage_execution.add_edge (postorder[0],
                                               stage_obj)
            logging.debug ("Stage '%s' created" % stage_name)
        else:
            stage_obj = nx.nodes (self.stage_execution).index (stage_obj)
            logging.warning ("Stage '%s' already exists in the stencil scope" % stage_name)
        return stage_obj


    def build_execution_path (self):
        """
        Analyzes the stages within this stencil scope and builds a graph
        representing their execution path
        :return:
        """
        leaves              = []
        new_stage_execution = nx.DiGraph ( )
        for stg in nx.topological_sort (self.stage_execution):
            new_stage_execution.add_node (stg)
            if stg.independent:
                #
                # adding independent stage
                #
                if len (leaves) == 1:
                    if not leaves[0].independent:
                        new_stage_execution.add_edge (leaves[0], stg)
                        leaves.clear ( )
                    else:
                        for pred in new_stage_execution.predecessors (leaves[0]):
                            new_stage_execution.add_edge (pred, stg)
                elif len (leaves) > 1:
                    for l in leaves:
                        assert (l.independent)
            else:
                #
                # adding NON independent stage
                #
                if len (leaves) > 0:
                    for l in leaves:
                        new_stage_execution.add_edge (l, stg)
                    leaves.clear ( )
            leaves.append (stg)
        #
        # save the newly built execution path
        #
        self.stage_execution = new_stage_execution
        #
        # ... and update the ghost-cell access pattern
        #
        self.update_ghost_cell ( )


    def check_stage_execution_path (self):
        """
        Runs various checks over the topology of the stage-execution graph
        :raise ValueError: if the last stage of the stencil is independent
        :return:
        """
        #
        # make sure the last stage is non-independent
        #
        correct = False
        for stg in nx.topological_sort (self.stage_execution,
                                        reverse=True):
            if not stg.independent:
                correct = correct or (len (self.stage_execution.successors (stg)) == 0)
        if correct:
            logging.info ("The stage-execution path looks valid")
        else:
            raise ValueError ("The last stage of stencil '%s' cannot be independent" % stencil.name)

    def runtime_analysis (self, stencil, **kwargs):
        """
        Attempts to aquire more information about the discovered symbols
        using runtime information of the user's stencil instance
        :param stencil:    the user's stencil instance
        :param kwargs:     the parameters passed to the stencil for execution
        :raise ValueError: if the last stage is independent, which is an invalid
                           stencil
        :return:
        """
        for s in self.get_all ( ):
            #
            # unresolved symbols have 'None' as their value
            #
            if s.value is None:
                #
                # is this a stencil's attribute?
                #
                if 'self' in s.name:
                    attr    = s.name.split ('.')[1]
                    s.value = getattr (stencil, attr, None)

                    #
                    # NumPy arrays are considered temporary data fields
                    #
                    if isinstance (s.value, np.ndarray):
                        #
                        # update the symbol table in this scope
                        #
                        self.add_temporary (s.name,
                                            s.value)
                    else:
                        self.add_constant (s.name,
                                           s.value)
        #
        # resolve the stencil parameters as scope symbols
        #
        self._resolve_params (stencil,
                              **kwargs)


    def update_ghost_cell (self):
        """
        Updates the ghost-cell access pattern of each stage of the stencil, the
        shape of which is based on the access patterns of their data fields
        :return:
        """
        all_stgs = nx.topological_sort (self.stage_execution,
                                        reverse=True)
        #
        # first, reset all ghost cells
        #
        for stg in all_stgs:
            stg.ghost_cell = None
        #
        # now start calculating them from the leaves ...
        #
        for stg in all_stgs:
            if len (self.stage_execution.successors (stg)) == 0:
                stg.ghost_cell = [0,0,0,0]
        #
        # ... towards the root
        #
        for stg in all_stgs:
            if stg.ghost_cell is None:
                succs = self.stage_execution.successors (stg)
                assert (len (succs) > 0)
                stg.ghost_cell = list (succs[0].ghost_cell)
                for suc in succs:
                    add_ghost = suc.scope.get_ghost_cell ( )
                    for idx in range (len (stg.ghost_cell)):
                        stg.ghost_cell[idx] += add_ghost[idx]
        if __debug__:
            for stg in all_stgs:
                logging.debug ("Stage '%s' has ghost cell %s" % (stg.name,
                                                                 stg.ghost_cell))
        #
        # calculate the minimum required halo
        #
        first_stg         = all_stgs[-1]
        self.minimum_halo = list (first_stg.ghost_cell)
        add_ghost         = first_stg.scope.get_ghost_cell ( )
        for idx in range (len (self.minimum_halo)):
            self.minimum_halo[idx] += add_ghost[idx]
        for idx in range (len (self.minimum_halo)):
            if self.minimum_halo[idx] < 0:
                self.minimum_halo[idx] *= -1
        logging.debug ("Minimum required halo is %s" % self.minimum_halo)

