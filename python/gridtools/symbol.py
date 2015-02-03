# -*- coding: utf-8 -*-
import logging
import numpy as np




class Symbol (object):
    """
    When traversing the AST of a user-generated stencil, the recognized
    symbols are instances of this class.-
    """
    kinds = (
        #
        # the 'param_rw' kind indicates a read-write stencil and/or functor 
        # data-field parameter
        #
        'param_rw',
        #
        # the 'param_ro' kind indicates a read-only stencil and/or functor 
        # data-field parameter
        #
        'param_ro',
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
            kind    a valid Symbol.kinds value;
            value   the value of the symbol.-
        """
        assert (kind in Symbol.kinds)
        self.name  = name
        self.kind  = kind
        self.value = value
        



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


    def add_symbol (self, name, kind, value=None):
        """
        Adds the received symbol to this scope:

            name        name of this symbol;
            kind        its kind (should be one of Symbol.kinds);
            value       its value.-
        """
        symbol = Symbol (name, kind, value)
        if name in self:
            logging.info ("Updated symbol '%s'" % name)
        self.symbol_table[name] = symbol


    def add_constant (self, name, value=None):
        """
        Adds a constant this scope:

            name    name of this constant;
            value   its value.-
        """
        if value:
            try:
                value = float (value)
                logging.info ("Constant '%s' has value %.3f" % (name,
                                                                value))
            except TypeError:
                if isinstance (value, np.ndarray):
                    logging.info ("Constant '%s' is a NumPy array %s" % (name,
                                                                         value.shape))
                else:
                    logging.info ("Constant '%s' has value %s" % (name,
                                                                  value))
        else:
            logging.info ("Constant '%s' will be resolved later" % name)
        self.add_symbol (name, 'const', value)


    def add_parameter (self, name, value=None, read_only=False):
        """ 
        Adds a parameter data field to this scope:

            name        the name of the parameter;
            value       its value;
            read_only   whether the parameter is read-only or not (default).-
        """
        if read_only:
            kind = 'param_ro'
        else:
            kind = 'param_rw'
        self.add_symbol (name, kind, value)
        if value is not None:
            if isinstance (value, np.ndarray):
                logging.info ("Parameter '%s' has dimension %s" % (name,
                                                                   value.shape))
            else:
                try:
                    logging.info ("Parameter '%s' has value %f.3f" % (name,
                                                                      float (value)))
                except ValueError:
                    logging.info ("Parameter '%s' has value %s" % (name,
                                                                   value))


    def add_temporary (self, name, value=None):
        """
        Adds a temporary data field to this scope:

            name    name of the temporary data field;
            value   its value (should be a NumPy array).-
        """
        if value:
            if isinstance (value, np.ndarray):
                #
                # add the field as a temporary
                #
                self.add_symbol (name, 'temp', value)
                logging.info ("Temporary field '%s' has dimension %s" % (name,
                                                                         value.dim))
            else:
                raise TypeError ("Temporary data field '%s' should be a NumPy array not '%s'" % 
                                 (type (value), name))
        else:
            raise ValueError ("Temporary data field '%s' cannot be None" % name)


    def arrange_ids (self):
        """
        Rearranges the IDs of all data fields in order to uniquely identify
        each of them.-
        """
        curr_id = 0
        for k in sorted (self.kinds, reverse=True):
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
            #
            # display only the dimensions of NumPy arrays
            #
            try:
                val = v.shape
            except AttributeError:
                val = str (v)

            logging.debug ("\t[%s]\t(%s)\t%s: %s" % (k, 
                                                     v.kind,
                                                     v.name,
                                                     val))


    def is_parameter (self, name, read_only=False):
        """
        Returns True is symbol 'name' is a parameter in this scope:

            read_only   whether the parameter is read-only or not (default).-
        """
        return name in [p.name for p in self.get_parameters (read_only)]


    def is_temporary (self, name):
        """
        Returns True if symbol 'name' is a temporary data field.-
        """
        return name in [t.name for t in self.get_temporaries ( )]


    def get_all (self):
        """
        Returns all symbols in this scope sorted by name.-
        """
        sorted_names = sorted (self.symbol_table.keys ( ))
        for n in sorted_names:
            yield self.symbol_table[n]


    def get_parameters (self, read_only=False):
        """
        Returns a sorted list of all parameters in this scope:

            read_only   whether the returned list should contain just read-only
                        parameters or not (default).-
        """
        kinds = ['param_ro']
        if not read_only:
            kinds.append ('param_rw')

        symbol_names = sorted (self.symbol_table.keys ( ))
        for n in symbol_names:
            if self.symbol_table[n].kind in kinds:
                yield self.symbol_table[n]


    def get_temporaries (self):
        """
        Returns a sorted list of all temporary data fields at this scope.-
        """
        symbol_names = sorted (self.symbol_table.keys ( ))
        for n in symbol_names:
            if self.symbol_table[n].kind == 'temp':
                yield self.symbol_table[n]




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


    def resolve (self, stencil_obj):
        """
        Attempts to aquire more information about the discovered symbols
        with runtime information of the stencil object 'stencil_obj'.-
        """
        #
        # we cannot change the symbol table while looping over it,
        # so we save the changes here and apply them afterwards
        #
        add_temps = dict ( )
        for s in self.stencil_scope.get_all ( ):
            #
            # unresolved symbols have 'None' as their value
            #
            if s.value is None:
                #
                # is this a stencil's attribute?
                #
                if 'self' in s.name:
                    attr  = s.name.split ('.')[1]
                    s.value = getattr (self, attr, None)

                    #
                    # NumPy arrays kept as stencil attributes are considered
                    # temporary data fields
                    #
                    if isinstance (s.value, np.ndarray):
                        #
                        # the new temporary data field will be added later,
                        # to prevent changes in the underlying data structure
                        # during the loop
                        #
                        add_temps[s.name] = s.value
                    else:
                        self.stencil_scope.add_constant (s.name, 
                                                         s.value)
            #
            # TODO some symbols are just aliases to other symbols
            #
            if isinstance (s.value, str):
                if s.value in symbols.keys ( ) and symbols[s.value] is not None:
                    #symbols[name] = symbols[value]
                    logging.warning ("Variable aliasing is not supported")

        #
        # update the symbol table now the loop has finished ...
        #
        for k,v in add_temps.items ( ):
            #
            # remove this field from the symbol table before adding it
            # as a temporary data field
            #
            del symbols[k]
            #
            # remove the 'self' from the attribute name
            #
            name = k.split ('.')[1]
            temp_field = FunctorParameter (name)
            temp_field.dim = v.shape
            symbols.add_temporary (name, temp_field)
        #
        # and rearrange all data fields IDs
        #
        #symbols.arrange_ids ( )

        #
        # print out the discovered symbols if in DEBUG mode
        #
        if __debug__:
            logging.debug ("Symbols found after using using run-time resolution:")
            self.stencil_scope.dump ( )
            for k,v in self.functor_scopes.items ( ):
                v.dump ( )


    def resolve_params (self, **kwargs):
        """
        Attempts to aquire more information about the discovered parameters 
        using runtime information.
        It returns the domain dimensions over which the stencil operates.-
        """
        domain = None
        for k,v in kwargs.items ( ):
            if self.stencil_scope.is_parameter (k):
                if isinstance (v, np.ndarray):
                    #
                    # update the value of this parameter
                    #
                    self.stencil_scope.add_parameter (k,
                                                      v,
                                                      read_only=self.stencil_scope.is_parameter (k,
                                                                                                 read_only=True))
                    #
                    #
                    # check the dimensions of different parameters match
                    #
                    if domain is None:
                        domain = v.shape
                    elif domain != v.shape:
                        logging.warning ("Dimensions of parameter '%s':%s do not match %s" % (k, v.shape, domain))
                else:
                    logging.warning ("Parameter '%s' is not a NumPy array" % k)
        return domain

