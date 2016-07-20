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
        'const',
        #
        # the 'local' kind indicates a data field that exists only within a stage
        #
        'local'
        )


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
        # an access extent defines the total data extent that needs to be
        # accessed for this symbol.
        # If the symbol object belongs to a StencilScope, this is the extent
        # required by the whole stencil.
        # If the symbol object belongs to a StageScope, this is the extent
        # required by the stage and considering all the stage successors.
        # Data format is the same as for the access pattern:
        #
        #   (minimum negative index accessed in _i_,
        #    maximum positive index accessed in _i_,
        #    minimum negative index accessed in _j_,
        #    maximum positive index accessed in _j_)
        #
        self.access_extent = None

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
        :raise IndexError: if the offset argument contains an invalid index
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
        Adds a alias to another known symbol
        :param name:                name of this symbol alias;
        :param known_symbol_name:   name of the known symbol this alias refers to.-
        :raise ValueError:          if the alias does not point to a known symbol
        """
        if known_symbol_name is not None:
            self.add_symbol (Symbol (name,
                                     'alias',
                                     known_symbol_name))
            logging.debug ("Alias '%s' refers to '%s'" % (name,
                                                          known_symbol_name))
        else:
            raise ValueError ("Alias '%s' does not point to any known symbol"
                              % name)


    def add_constant (self, name, value=None):
        """
        Adds a constant this scope
        :param name:    name of this constant;
        :param value:   its value.-
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


    def add_local (self, name, value=None):
        """
        Adds a local variable to this scope
        :param name:        the name of the local variable
        :param value:       value of the variable
        """
        self.add_symbol (Symbol (name, 'local', value))
        if value is None:
            logging.debug ("Local variable '%s' is None" % name)
        else:
            logging.debug ("Local variable '%s' has value %s" % (name, value))



    def add_parameter (self, name, value=None, read_only=True):
        """
        Adds a parameter data field to this scope
        :param name:        the name of the parameter
        :param value:       its value
        :param read_only:   whether the parameter is read-only (default)
        :return:
        """
        if name not in self:
            self.add_symbol (Symbol (name, 'param', value))
        self[name].kind      = 'param'
        self[name].read_only = read_only
        if value is not None:
            self[name].value = value
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
        Adds or updates the received symbol in this scope
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
        Returns True if symbol with 'name' is an alias.-
        """
        return name in [a.name for a in self.get_aliases ( )]


    def is_constant (self, name):
        """
        Returns True if symbol with 'name' is a constant.-
        """
        return name in [c.name for c in self.get_constants ( )]


    def is_local (self, name):
        """
        Returns True if symbol 'name' is a local variable
        """
        return name in [l.name for l in self.get_locals ( )]


    def is_parameter (self, name):
        """
        Returns True if symbol 'name' is a parameter in this scope.-
        """
        return name in [p.name for p in self.get_parameters ( )]


    def is_temporary (self, name):
        """
        Returns True if symbol 'name' is a temporary data field.-
        """
        return name in [t.name for t in self.get_temporaries ( )]


    def get_all (self, kinds=None):
        """
        Returns a generator to all symbols in this scope sorted by name:

        :param kinds:   returns only symbol kinds contained in this list.-
        """
        sorted_names = sorted (self.symbol_table.keys ( ))
        if kinds is None:
            for n in sorted_names:
                yield self.symbol_table[n]
        else:
            for n in sorted_names:
                if self.symbol_table[n].kind in kinds:
                    yield self.symbol_table[n]


    def get_aliases (self):
        """
        Returns a sorted list of all aliases in this scope.-
        """
        return self.get_all (['alias'])


    def get_constants (self):
        """
        Returns a sorted list of all constants in this scope.-
        """
        return self.get_all (['const'])


    def get_locals (self):
        """
        Returns a sorted list of all local variables in this scope.-
        """
        return self.get_all (['local'])


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


    def minimum_enclosing_extent (self, ext1, ext2):
        """
        Returns the minimum enclosing extent of two extents (i.e. intervals).
        The data format is the same of halos and symbol access patterns.
        """
        mee = [None,None,None,None]

        mee[0] = min (ext1[0], ext2[0])
        mee[1] = max (ext1[1], ext2[1])
        mee[2] = min (ext1[2], ext2[2])
        mee[3] = max (ext1[3], ext2[3])

        return mee


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
        #
        # kernel line number in the self.py_src code
        #
        self.kernel_lineno   = None


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


    def check_self_dependent_fields (self):
        """
        Verifies that the stencil's data dependencies comply with Gridtools'
        rules. For more information, see
        https://github.com/eth-cscs/gridtools/wiki/Data-Dependencies-Analysis-in-GridTools
        :raises ValueError: if a data field depends on itself, either directly
                            of indirectly
        :returns:
        """
        #
        # A data field is depending on itself if the intersection of its
        # ancestors and it descendants is not empty. This means that a closed
        # path exists within the graph.
        # A self-dependency exists also if a data filed is assigned to itself
        # inside a stage, but this case should already be covered when
        # identifying stage IO.
        #
        # TODO: Only allow self-dependency if access extent is [0,0], complying
        # with Gridtools' data dependency rules.
        #
        for node in self.data_dependency.nodes_iter():
            closed_path = ( nx.ancestors(self.data_dependency, node)
                            & nx.descendants(self.data_dependency, node) )
            if closed_path:
                raise ValueError ("A stencil data field cannot depend on itself.")


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
        logging.debug ('Adding stage: %s' % stage_name)

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


    def build_execution_path (self, stencil_name):
        """
        Analyzes the stages within this stencil scope and builds a graph
        representing their execution path

        After the StencilInspector has finished its job, the StencilScope
        contains a preliminary stage execution graph, formed by a linear chain
        of stages in the order they were entered in the stencil kernel.

        We can iteratively build the final and correct stage execution graph by
        traversing the preliminary graph: for each node, we add it in the new
        graph, and look if any input of the current node matches an output on
        other nodes of the new graph. This assumes that IO identification for
        all stages has already been carried out.

        When introducing a node in the new execution graph, all the stages
        already present in the graph are at the same or higher level with
        respect of the current node (due to the order of the preliminary graph).
        Thus, we automatically avoid a wrong link to a stage that should depend
        on the introduced one (because no such stage is present in the graph).

        To avoid duplicating connections, a given stage input is removed from the
        available ones after a match is found from a postorder topological sort
        of the new execution graph. The sort is executed in postorder to link to
        the latest possible stage between those having the same output data
        field.
        TODO: Inform the user of this policy in the proper documentation.
        :param stencil_name:    The name of the stencil owning this scope
        :return:
        """
        logging.debug("Building final stage execution path for stencil %s" %
                      stencil_name)
        new_stage_execution = nx.DiGraph ( )
        for stg in nx.topological_sort (self.stage_execution):
            logging.debug('Processing stage: %s' % stg.name)
            new_stage_execution.add_node(stg)
            #
            # Create modifiable lists for stage input and output
            #
            stg_in = list(stg.inputs)
            #
            # Look for connections among nodes introduced in the new graph
            # Use the reverse sort order to find the latest nodes with the
            # desired output.
            #
            for other in nx.topological_sort(new_stage_execution, reverse=True):
                logging.debug('\tLooping other nodes: %s' % other.name)
                #
                # Skip self (obviously...)
                #
                if other is stg:
                    continue
                for i, data in enumerate(stg_in):
                    if data in other.outputs:
                        #
                        # We have found a match for this input.
                        # If the stage we are processing has no connections,
                        # create a new edge
                        #
                        logging.debug('\t\tMatch found!')
                        if not new_stage_execution.predecessors (stg):
                            new_stage_execution.add_edge(other, stg)
                        #
                        # Else, check that the "other" node is not an ancestor
                        # of the processed stage. This avoids duplicate and wrong
                        # connections between nodes.
                        #
                        else:
                            for pred in new_stage_execution.predecessors (stg):
                                if other not in nx.ancestors(new_stage_execution, pred):
                                    new_stage_execution.add_edge(other, stg)
        #
        # save the newly built execution path
        #
        self.stage_execution = new_stage_execution


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


    def compute_access_extents (self):
        """
        Compute the access extents of all parameters in each stage scope and at
        stencil scope.

        The algorithm starts from the leaves of the stage execution graph and
        progressively refines the access extents by visiting stages with a
        postorder traversal from each leaf.

        When visiting a stage, the extent of a parameter is updated taking into
        account its access pattern in that stage (access patterns are set during
        code generation), its access extent in the successor stages (which in
        turn represents the aggregation of all access extents for that symbol
        starting from the leaves), and the access extent of the stage outputs that
        depend on the parameter, if those outputs are accessed from a stage
        succeeding the currently visited one.

        Working on parameters assures that the same symbol can be found in
        different stages using the same name, because parameters are first and
        foremost stencil scope symbols, which are passed on to stages.

        When all traversals have been completed, the access extents of
        parameters from the stages are combined to yield
        the access extents at stencil scope.
        """
        stages = nx.topological_sort (self.stage_execution,
                                      reverse=True)
        #
        # Initialize access extents on leaf stages; create list of non-leaf stages
        #
        non_leaf_stages = []
        for stg in stages:
            #
            # Leaf stages
            #
            if len (self.stage_execution.successors (stg)) == 0:
                #
                # Access extents are equal to access patterns in leaf stages
                #
                for par in stg.scope.get_parameters ( ):
                    if par.access_pattern is not None:
                        par.access_extent = list(par.access_pattern)
            else:
                non_leaf_stages.append (stg)

        for stg in non_leaf_stages:
            succs = self.stage_execution.successors (stg)
            assert (len (succs) > 0)
            for suc in succs:
                #
                # Loop over parameters
                #
                for par in stg.scope.get_parameters ( ):
                    #
                    # To compute the extent for this parameter we have to find
                    # the minimum enclosing scope between the parameter's access
                    # pattern in this stage and the sum of the access extent in
                    # the successor plus the extent in the successor of any
                    # symbol dependent on the current parameter. In short:
                    #
                    # extent = mee(access_pattern, extent_in_successor + dependant_symbol_extent_in_successor)
                    #
                    # First, initialize successor extent and the dependencies
                    # extents...
                    #
                    successor_extent     = None
                    dependencies_extents = []
                    #
                    # Get the extent of this parameter in the successor as a
                    # numpy array
                    #
                    if suc.scope.is_parameter (par.name):
                        suc_ext = suc.scope.symbol_table[par.name].access_extent
                        if suc_ext is not None:
                            successor_extent = np.array (suc_ext, dtype=np.int)
                    for output in stg.outputs:
                        if output == par:
                            continue

                        if output in stg.scope.data_dependency:
                            dd_node = output
                        else:
                            #
                            # Output is not in data depencency graph for
                            # this stage. Look for its alias.
                            #
                            for alias in stg.scope.get_aliases ( ):
                                if alias.value == output:
                                    dd_node = alias
                        #
                        # If a stage output depends from the current parameter...
                        #
                        for des in nx.descendants (stg.scope.data_dependency, dd_node):
                            if stg.scope.is_alias (des):
                                des = stg.scope.symbol_table[des.value]
                            if des.name == par.name:
                                #
                                # ...and if the output is an input of the successor...
                                #
                                if output.name in [i.name for i in suc.inputs]:
                                    #
                                    # ...get the access extent of the successor's input (finally!)
                                    #
                                    dep_ext = suc.scope.symbol_table[output.name].access_extent
                                    if dep_ext is not None:
                                        dependencies_extents.append (np.array (dep_ext, dtype=np.int))
                                    else:
                                        dependencies_extents.append (np.zeros (4, dtype=np.int))
                    #
                    # Verify we have all 3 elements necessary to compute the
                    # minimum enclosing extent (mee). If we only some of them
                    # are available, initialize the others to [0,0,0,0] so they
                    # will be neutral elements in the operation.
                    #
                    mee_elements = [successor_extent,
                                    par.access_pattern,
                                    dependencies_extents]
                    if any ([len(x)!=0 for x in mee_elements if x is not None]):
                        if successor_extent is not None:
                            mee = successor_extent
                        else:
                            mee = np.zeros (4, dtype=np.int)

                        if par.access_pattern is not None:
                            acc_patt = par.access_pattern
                        else:
                            acc_patt = np.zeros (4, dtype=np.int)

                        if not dependencies_extents:
                            dependencies_extents.append (np.zeros(4, dtype=np.int))
                        #
                        # Now we can find the minimum enclosing extent for the
                        # parameter, starting from the extent in the successor
                        #
                        for dep_ext in dependencies_extents:
                            mee = self.minimum_enclosing_extent (mee, dep_ext + acc_patt)
                        #
                        # Set the parameter access extent.
                        # Do a minimum enclosing scope operation once again if
                        # access extent has already been initialized.
                        #
                        if par.access_extent is None:
                            par.access_extent = mee
                        else:
                            par.access_extent = self.minimum_enclosing_extent (par.access_extent,
                                                                           mee)
        #
        # Aggregate stages' access extents at stencil scope
        #
        from itertools import chain
        partemps = chain (self.get_parameters ( ), self.get_temporaries ( ))
        for pt in partemps:
            for stg in stages:
                if stg.scope.is_parameter (pt.name):
                    stage_ext = stg.scope.symbol_table[pt.name].access_extent
                    if stage_ext is not None:
                        if pt.access_extent is None:
                            pt.access_extent = [0,0,0,0]
                        pt.access_extent = self.minimum_enclosing_extent (pt.access_extent,
                                                                          stage_ext)


    def compute_minimum_halo (self):
        """
        Compute stencil's minimum halo by combining the access extents of stencil
        parameters.
        """
        self.minimum_halo = [0,0,0,0]
        for par in self.get_parameters ( ):
            if par.access_extent is not None:
                self.minimum_halo = self.minimum_enclosing_extent (self.minimum_halo,
                                                                   par.access_extent)
        self.minimum_halo[0] *= -1
        self.minimum_halo[2] *= -1
        logging.debug ("Minimum required halo is %s" % self.minimum_halo)


    def update_ghost_cell (self):
        """
        Updates the ghost-cell access pattern of each stage of the stencil, the
        shape of which is based on the access patterns of their data fields

        :return:
        """
        all_stgs = nx.topological_sort (self.stage_execution,
                                        reverse=True)
        for stg in all_stgs:
            #
            # Initialize the ghost cell
            #
            stg.ghost_cell  = [0,0,0,0]
            #
            # Stage ghost cell is the minimum enclosing extent of its output,
            # where the output extent is the one at stencil scope
            #
            for out in stg.outputs:
                out_ext = self.symbol_table[out.name].access_extent
                if out_ext is None:
                    out_ext  = [0,0,0,0]
                stg.ghost_cell = self.minimum_enclosing_extent (stg.ghost_cell,
                                                                out_ext)
            #
            # Convert all numbers in the ghost cell descriptor to positive,
            # making it compliant to the same logic of the minimum halo
            # descriptor
            #
            stg.ghost_cell[0] *= -1
            stg.ghost_cell[2] *= -1

            logging.debug ("Stage '%s' has ghost cell %s" % (stg.name,
                                                             stg.ghost_cell))
