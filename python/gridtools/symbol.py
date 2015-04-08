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
        # the 'param' kind indicates a stencil and/or functor data-field parameter
        #
        'param',
        #
        # the 'temp' kind indicates a temporary data field
        #
        'temp',
        #
        # the 'local' kind indicates a variable that is defined in the current
        # scope, i.e., only visible to the current and inner scopes 
        #
        'local',
        #
        # the 'const' kind indicates a variable that may POTENTIALLY be
        # replaced by a value. This may happen by inlining them of using 
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
        # a range defines the extent of data accessed during stencil execution;
        # its structure is defined like this:
        #
        #   (minimum index accessed in _i_, maximum index accessed in _i_, 
        #    minimum index accessed in _j_, maximum index accessed in _j_)
        #
        self.range = None

        #
        # a flag indicating whether this symbol is read-only
        #
        self.read_only = True
        

    def __hash__ (self):
        return self.name.__hash__ ( )

    def __repr__ (self):
        return self.name


    def set_range (self, rng):
        """
        Sets or updates the range of this symbol.-
        """
        try:
            i, j, k = rng

            if self.range is None:
                self.range = [0, 0, 0, 0]

            if self.range[0] > i:
                self.range[0] = i
            elif self.range[1] < i:
                self.range[1] = i

            if self.range[2] > j:
                self.range[2] = j
            elif self.range[3] < j:
                self.range[3] = j

        except IndexError:
            logging.error ("Range descriptor %s should be 3-dimensional" % rng)



class SymbolInspector (ast.NodeVisitor):
    """
    Inspects the AST looking for known symbols, keeping their dependency graph.-
    """
    def __init__ (self, scope):
        self.scope         = scope
        self.symbols_found = None


    def search (self, node):
        """
        Returns a list of the symbols belonging to the current scope that are
        found in the AST, the root of which is 'node'.-
        """
        self.symbols_found = list ( )
        self.visit (node)
        return self.symbols_found
        

    def visit_Attribute (self, node):
        """
        Looks for this attribute in the current scope.-
        """
        name = "%s.%s" % (node.value.id,
                          node.attr)
        if name in self.scope:
            self.symbols_found.append (self.scope[name])


    def visit_Name (self, node):
        """
        Looks for this name in the current scope.-
        """
        name = node.id
        if name in self.scope:
            self.symbols_found.append (self.scope[name])


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
        self.depency_graph = nx.DiGraph ( )


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
        Creates a dependency between 'left_symbol' and 'right_symbol'.-
        """
        if (left_symbol.name in self) and (right_symbol.name in self):
            self.depency_graph.add_edge (left_symbol,
                                         right_symbol)
        else:
            raise ValueError ("Trying to add a dependency between non-existent symbols")


    def add_dependencies (self, deps):
        """
        Adds a sequence of data dependencies, each of which must be given
        as a 2-tuples (u,v).-
        """
        for d in deps:
            self.add_dependency (d[0], d[1])


    def add_parameter (self, name, value=None, read_only=True):
        """ 
        Adds a parameter data field to this scope:

            name        the name of the parameter;
            value       its value;
            read_only   whether the parameter is read-only (default).-
        """
        self.add_symbol (Symbol (name, 'param', value))
        self[name].read_only = read_only
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
        Adds a temporary data field to this scope:

            name    name of the temporary data field;
            value   its value (should be a NumPy array).-
        """
        if value is not None:
            if isinstance (value, np.ndarray):
                #
                # add the field as a temporary
                #
                self.add_symbol (Symbol (name, 'temp', value))
                logging.debug ("Temporary field '%s' has dimension %s" % (name,
                                                                         value.shape))
            else:
                raise TypeError ("Temporary data field '%s' should be a NumPy array not '%s'" % 
                                 (type (value), name))
        else:
            raise ValueError ("Temporary data field '%s' cannot be None" % name)


    def add_symbol (self, symbol):
        """
        Adds the received 'symbol' to this scope.-
        """
        if symbol.name in self:
            logging.debug ("Updated symbol '%s'" % symbol.name)
        self.symbol_table[symbol.name] = symbol


    def arrange_ids (self):
        """
        Rearranges the IDs of all data fields in order to uniquely identify
        each of them.-
        """
        curr_id = 0
        for k in sorted (Symbol.KINDS, reverse=True):
            for k,v in self.symbol_table[k].items ( ):
                try:
                    v.id = curr_id
                    curr_id += 1
                except AttributeError:
                    #
                    # not a data field, ignore it
                    #
                    pass


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


    def is_constant (self, name):
        """ 
        Returns True if symbol 'name' is a constant.-
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



class StencilSymbols (object):
    """
    Stencil symbols are organized into scopes that alter the visibility 
    of the variables defined in the stencil or any of its functors.
    Basically, if a symbol is defined at stencil level, the stencil itself and
    all its functors can access the it. If it is defined at functor level, it
    is accesible within that functor only.-
    """
    def __init__ (self):
        #
        # the top-most stencil scope
        #
        self.stencil_scope = Scope ( )

        #
        # the scope of each stencil functor is kept as a dict, i.e.:
        #
        #       A = (k=functor name, v=scope)
        #
        self.functor_scopes = dict ( )


    def add_functor (self, funct_name):
        """
        Returns a new scope for keeping the functor's symbols:

            funct_name  a unique name for the functor.-
        """
        if funct_name in self.functor_scopes.keys ( ):
            raise NameError ("Functor '%s' already exists in symbol table.-" % funct_name)
        else:
            self.functor_scopes[funct_name] = Scope ( )
            return self.functor_scopes[funct_name]


