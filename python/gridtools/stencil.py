# -*- coding: utf-8 -*-
import logging

import numpy as np
import networkx as nx

from gridtools.symbol   import StencilScope
from gridtools.functor  import Functor
from gridtools.compiler import StencilCompiler




class InteriorPoint (tuple):
    """
    Represents the point within a NumPy array at the given coordinates.-
    """
    def __add__ (self, other):
        if len (self) != len (other):
            raise ValueError ("Points have different dimensions.")
        return tuple (map (sum, zip (self, other)))

    def __sub__ (self, other):
        raise NotImplementedError ("Offsets with '-' are not supported.")



class Stencil (object):
    """
    A base stencil class.-
    """
    #
    # a JIT compiler class shared by all stencils
    #
    compiler = StencilCompiler ( )

    def __init__ (self):
        #
        # register this stencil with the compiler and inspector
        #
        self.name        = Stencil.compiler.register (self)
        #
        # defines the way to execute the stencil
        #
        self._backend    = "python"
        #
        # the domain dimensions over which this stencil operates
        #
        self.domain      = None
        #
        # symbols gathered after analyzing the stencil code are kept here
        #
        self.scope       = StencilScope ( )
        #
        # a halo descriptor
        #
        self.halo        = (0, 0, 0, 0)
        #
        # define the execution order in 'k' dimension
        #
        self.k_direction = 'forward'


    def __copy__ (self, memo):
        raise NotImplementedError ("Cannot create shallow copies of a Stencil")


    def __deepcopy__ (self, memo):
        from copy import deepcopy

        cls            = self.__class__
        rv             = cls.__new__ (cls)
        memo[id(self)] = rv
        for k, v in self.__dict__.items ( ):
            #
            # deep copy all the attributes, but the stencil scope
            #
            if k != 'scope':
                setattr (rv, k, deepcopy (v, memo))
        setattr (rv, 'scope', StencilScope ( ))
        return rv


    def __hash__ (self):
        return self.name.__hash__ ( )


    def __repr__ (self):
        return self.name


    def _plot_graph (self, G):
        """
        Renders graph 'G' using 'matplotlib'.-
        """
        from gridtools import plt

        pos = nx.spring_layout (G)
        nx.draw_networkx_nodes (G,
                                node_size=1500,
                                pos=pos)
        nx.draw_networkx_edges (G,
                                pos=pos,
                                arrows=True)
        nx.draw_networkx_labels (G,
                                 pos=pos)


    @property
    def backend (self):
        return self._backend


    @backend.setter
    def backend (self, value):
        self._backend = value
        Stencil.compiler.recompile (self)


    def generate_code (self):
        """
        Generates C++ code for this stencil
        :raise Exception: in case of an error during the code-generation process
        :return:
        """
        #
        # generate the code of *all* stages in this stencil,
        # building a data-dependency graph among their data fields
        #
        for stg in self.stages:
            stg.generate_code           ( )
            self.scope.add_dependencies (stg.get_data_dependency ( ).edges ( ))


    def get_interior_points (self, data_field, ghost_cell=[0,0,0,0]):
        """
        Returns an iterator over the 'data_field' without including the halo:

            data_field      a NumPy array;
            ghost_cell      access pattern for the current field, which depends
                            on the following stencil stages.-
        """
        try:
            if len (data_field.shape) != 3:
                raise ValueError ("Only 3D arrays are supported.")

        except AttributeError:
            raise TypeError ("Calling 'get_interior_points' without a NumPy array")
        else:
            #
            # calculate 'i','j','k' iteration boundaries
            # based on 'halo' and field-access patterns
            #
            i_dim, j_dim, k_dim = data_field.shape

            start_i = 0     + self.halo[0] + ghost_cell[0]
            end_i   = i_dim - self.halo[1] + ghost_cell[1]
            start_j = 0     + self.halo[2] + ghost_cell[2]
            end_j   = j_dim - self.halo[3] + ghost_cell[3]

            #
            # calculate 'k' iteration boundaries based 'k_direction'
            #
            if self.k_direction == 'forward':
                start_k = 0
                end_k   = k_dim
                inc_k   = 1
            elif self.k_direction == 'backward':
                start_k = k_dim - 1
                end_k   = -1
                inc_k   = -1
            else:
                logging.warning ("Ignoring unknown K direction '%s'" % self.k_direction)
            #
            # return the coordinate tuples in the correct order
            #
            for i in range (start_i, end_i):
                for j in range (start_j, end_j):
                    for k in range (start_k, end_k, inc_k):
                        yield InteriorPoint ((i, j, k))


    def kernel (self, *args, **kwargs):
        """
        This function is the entry point of the stencil and
        should be implemented by the user.-
        """
        raise NotImplementedError ( )


    def plot_3d (self, Z):
        """
        Plots the Z field in 3D, returning a Matplotlib's Line3DCollection.-
        """
        if len (Z.shape) == 2:
            from gridtools import plt

            fig = plt.figure ( )
            ax  = fig.add_subplot (111,
                                   projection='3d',
                                   autoscale_on=True)
            X, Y = np.meshgrid (np.arange (Z.shape[0]),
                                np.arange (Z.shape[1]))
            im = ax.plot_wireframe (X, Y, Z,
                                    linewidth=1)
            return im
        else:
            logging.error ("The passed Z field should be 2D")


    def plot_data_dependency (self, graph=None):
        """
        Renders a data-depencency graph using 'matplotlib'
        :param graph: the graph to render; it renders this stencil's data
                      dependency graph if None given
        :return:
        """
        if graph is None:
            graph = self.scope.data_dependency
        self._plot_graph (graph)


    def run (self, *args, **kwargs):
        """
        Starts the execution of the stencil
        :raise KeyError:   if any non-keyword arguments were passed
        :raise ValueError: if the backend is not recognized or if the halo is
                           invalid
        :return:
        """
        #
        # we only accept keyword arguments to avoid confusion
        #
        if len (args) > 0:
            raise KeyError ("Only keyword arguments are accepted")
        #
        # make sure the stencil is registered with the compiler
        #
        if not Stencil.compiler.is_registered (self):
            Stencil.compiler.register (self)
        #
        # analyze the stencil code
        #
        try:
            Stencil.compiler.analyze (self, **kwargs)
        except Exception as e:
            logging.error("Error while analyzing code for stencil '%s'" % self.name)
            Stencil.compiler.unregister (self)
            raise e
        else:
            #
            # check the minimum halo has been given
            #
            for idx in range (len (self.scope.minimum_halo)):
                if self.scope.minimum_halo[idx] - self.halo[idx] > 0:
                    raise ValueError ("The halo should be at least %s" %
                                      self.scope.minimum_halo)
            #
            # run the selected backend version
            #
            logging.info ("Executing '%s' in %s mode ..." % (self.name,
                                                             self.backend.upper ( )))
            if self.backend == 'c++' or self.backend == 'cuda':
                Stencil.compiler.run_native (self, **kwargs)
            elif self.backend == 'python':
                self.kernel (**kwargs)
            else:
                raise ValueError ("Unknown backend '%s'" % self.backend)


    def set_halo (self, halo=(0,0,0,0)):
        """
        Applies the received 'halo' setting, which is defined as

            (halo in negative direction over _i_,
             halo in positive direction over _i_,
             halo in negative direction over _j_,
             halo in positive direction over _j_).-
        """
        if halo is None:
            return
        if len (halo) == 4:
            if halo[0] >= 0 and halo[2] >= 0:
                if halo[1] >= 0 and halo[3] >= 0:
                    self.halo = halo
                    Stencil.compiler.recompile (self)
                    logging.debug ("Setting halo to %s" % str (self.halo))
                else:
                    raise ValueError ("Invalid halo %s: definition for the positive halo should be zero or a positive integer" % str (halo))
            else:
                raise ValueError ("Invalid halo %s: definition for the negative halo should be zero or a positive integer" % str (halo))
        else:
            raise ValueError ("Invalid halo %s: it should contain four values" % str (halo))


    def set_k_direction (self, direction="forward"):
        """
        Applies the execution order in `k` dimension:

            direction   defines the execution order, which may be any of:
                        forward or backward.-
        """
        accepted_directions = ["forward", "backward"]

        if direction is None:
            return

        if direction in accepted_directions:
            self.k_direction = direction
            Stencil.compiler.recompile (self)
            logging.debug ("Setting k_direction to '%s'" % self.k_direction)
        else:
            logging.warning ("Ignoring unknown direction '%s'" % direction)


    @property
    def stages (self):
        return nx.topological_sort (self.scope.stage_execution)



class MultiStageStencil (Stencil):
    """
    A involving several stages, implemented as for-comprehensions.
    All user-defined stencils should inherit for this class.-
    """
    def __init__ (self):
        super ( ).__init__ ( )


    def build (self, output, **kwargs):
        """
        Use this stencil as part of a new CombinedStencil, the output parameter
        of which is called 'output' and should be linked to '**kwargs' parameters
        of the following stencil.-
        """
        #
        # make sure the output parameter is there
        #
        if output is None:
            raise ValueError ("You must specify an 'output' parameter")
        #
        # the keyword arguments map the output of this stencil
        # with an input parameter of the following one
        #
        ret_value = CombinedStencil ( )
        ret_value.add_stencil (self,
                               output=output,
                               **kwargs)
        return ret_value



class CombinedStencil (Stencil):
    """
    A stencil created as a combination of several MultiStageStencils.-
    """
    def __init__ (self):
        """
        Creates a CombinedStencil with an output field named 'output_field'.-
        """
        super ( ).__init__ ( )
        #
        # graphs for data-dependency and execution order
        #
        self.data_graph      = nx.DiGraph ( )
        self.execution_graph = nx.DiGraph ( )


    def _extract_domain (self, **kwargs):
        """
        Returns the current domain dimensions from 'kwargs'.-
        """
        domain = None
        for k,v in kwargs.items ( ):
            try:
                if domain is None:
                    domain = v.shape
                if domain != v.shape:
                    logging.warning ("Different domain sizes detected")
            except AttributeError:
                #
                # not a NumPy array
                #
                pass
        if domain is None:
            raise RuntimeError ("Could not infer data-field dimensions")
        return domain


    def _prepare_parameters (self, stencil, **kwargs):
        """
        Extracts the parameters for 'stencil' from 'kwargs', and create
        intermediate data fields for linked stencils.-
        """
        ret_value = dict ( )
        domain    = self._extract_domain (**kwargs)
        for p in stencil.scope.get_parameters ( ):
            try:
                ret_value[p.name] = kwargs[p.name]
            except KeyError:
                #
                # get the value of this parameter from the data-dependency graph
                #
                if self.data_graph.has_node (p.name):
                    linked_data = self.data_graph.successors (p.name)
                    if len (linked_data) == 0:
                        #
                        # create a new linked data field
                        #
                        self.data_graph.node[p.name]['value'] = np.zeros (domain)
                        ret_value[p.name] = self.data_graph.node[p.name]['value']
                    elif len (linked_data) == 1:
                        #
                        # get the value from the linked field
                        #
                        ret_value[p.name] = self.data_graph.node[linked_data[0]]['value']
                else:
                    raise RuntimeError ("Parameter '%s' not found in data graph"
                                        % p.name)
        return ret_value


    def _update_parameters (self):
        """
        Updates the parameters of this combined stencil.-
        """
        #
        # add all the parameters of all the stencils combined
        #
        for stencil in self.execution_graph.nodes_iter ( ):
            for p in stencil.scope.get_parameters ( ):
                self.scope.add_parameter (p.name,
                                          p.value,
                                          p.read_only)
        #
        # remove linked parameters, i.e., output of one stencil is the input
        # of the following one
        #
        for n in self.data_graph.nodes_iter ( ):
            try:
                self.scope.remove (n)
            except KeyError:
                pass


    def add_stencil (self, stencil, output, **kwargs):
        """
        Adds a stencil, the output of which is called 'output' and should be
        forwarded into an input parameter of an adjacent stencil in the
        execution graph.-
        """
        try:
            out_param = stencil.scope[output]
            if stencil.scope.is_parameter (out_param.name):
                #
                # add 'stencil' to the execution graph
                #
                stencil_params = [p for p in stencil.scope.get_parameters ( )]
                self.execution_graph.add_node (stencil,
                                               output=out_param.name,
                                               params=stencil_params)
                if len (kwargs) > 0:
                    #
                    # 'stencil' is linked to the output of other stencils
                    #
                    for input_param in kwargs.keys ( ):
                        input_stencil = kwargs[input_param]
                        try:
                            #
                            # update the execution graph ...
                            #
                            for n,d in input_stencil.execution_graph.nodes_iter (data=True):
                                self.execution_graph.add_node (n, d)
                            for u,v in input_stencil.execution_graph.edges_iter ( ):
                                self.execution_graph.add_edge (u, v)
                            #
                            # ... and the data graph of this stencil
                            #
                            for n,d in input_stencil.data_graph.nodes_iter (data=True):
                                self.data_graph.add_node (n, d)
                            for u,v in input_stencil.data_graph.edges_iter ( ):
                                self.data_graph.add_edge (u, v)
                            #
                            #
                            # link the data dependency with the other stencil
                            #
                            linked_stencil        = input_stencil.get_root ( )
                            linked_stencil_output = input_stencil.execution_graph.node[linked_stencil]['output']
                            self.data_graph.add_node (input_param,
                                                      {'value': None})
                            self.data_graph.add_node (linked_stencil_output,
                                                      {'value': None})
                            self.data_graph.add_edge (input_param,
                                                      linked_stencil_output)
                            #
                            # ... and the execution dependency
                            #
                            self.execution_graph.add_edge (stencil,
                                                           linked_stencil)
                        except AttributeError:
                            logging.error ("Parameter '%s' should hold an instance of '%s'"
                                           % (input_param, self.__class__))
                            return
                #
                # update the parameters of this combined stencil
                #
                self._update_parameters ( )
                logging.debug ("Execution graph of '%s'\n\t%s" % (self,
                                                                  self.execution_graph.edges (data=True)))
            else:
                raise ValueError ("'%s' is not a parameter of '%s'" % (output,
                                                                       stencil.name))
        except KeyError:
            raise ValueError ("'%s' is not a parameter of '%s'" % (output,
                                                                   stencil.name))


    def generate_code (self, src_dir=None):
        """
        Generates native code for this stencil:

            src_dir     directory where the files should be saved (optional).-
        """
        from os        import write, path, makedirs
        from tempfile  import mkdtemp
        from gridtools import JinjaEnv

        #
        # create directory and files for the generated code
        #
        if src_dir is None:
            self.src_dir = mkdtemp (prefix="__gridtools_")
        else:
            if not path.exists (src_dir):
                makedirs (src_dir)
            self.src_dir = src_dir

        functor_src       = ''
        self.lib_file     = 'lib%s' % self.name.lower ( )
        self.cpp_file     = '%s.cpp' % self.name
        self.make_file    = 'Makefile'
        self.fun_hdr_file = '%sFunctors.h' % self.name

        logging.info ("Generating C++ code in '%s'" % self.src_dir)
        #
        # generate the code for *all* functors within the combined stencil
        #
        for st in nx.dfs_postorder_nodes (self.execution_graph,
                                          source=self.get_root ( )):
            for func in st.inspector.functors:
                func.generate_code (st.inspector.src)
                st.scope.add_dependencies (func.get_data_dependency ( ).edges ( ))
            fun_src, _, _  = Stencil.compiler.translate (st)
            functor_src   += fun_src

        with open (path.join (self.src_dir, self.fun_hdr_file), 'w') as fun_hdl:
            functors = JinjaEnv.get_template ("functors.h")
            fun_hdl.write (functors.render (functor_src=functor_src))
        #
        # code for the stencil, the library entry point and makefile
        #
        cpp_src, make_src = self.translate ( )
        with open (path.join (self.src_dir, self.cpp_file), 'w') as cpp_hdl:
            cpp_hdl.write (cpp_src)
        with open (path.join (self.src_dir, self.make_file), 'w') as make_hdl:
            make_hdl.write (make_src)


    def get_root (self):
        """
        Returns the root node of the execution graph, i.e., the *last* stencil
        to be executed, including its associated data.-
        """
        ret_value = None
        for node in self.execution_graph.nodes_iter ( ):
            if len (self.execution_graph.predecessors (node)) == 0:
                if ret_value is None:
                    ret_value = node
                else:
                    raise RuntimeError ("The execution graph of '%s' contains two roots"
                                        % self)
        if ret_value is None:
            logging.warning ("Root node could not be found")
            ret_value = self.execution_graph.nodes ( )[0]
        return ret_value


    def plot_data_graph (self):
        """
        Renders the data graph using 'matplotlib'.-
        """
        self._plot_graph (self.data_graph)


    def plot_execution_graph (self):
        """
        Renders the execution graph using 'matplotlib'.-
        """
        self._plot_graph (self.execution_graph)


    def resolve (self, **kwargs):
        #
        # resolution order is from the leafs towards the root
        #
        for st in nx.dfs_postorder_nodes (self.execution_graph,
                                          source=self.get_root ( )):
            st.backend = self.backend
            params     = self._prepare_parameters (st,
                                                   **kwargs)
            st.resolve (**params)
            #
            # update the current field domain
            #
            if self.inspector.domain is not None:
                if self.inspector.domain != st.inspector.domain:
                    logging.warning ("Different domain sizes detected")
            self.inspector.domain = st.inspector.domain


    def run (self, *args, halo=None, k_direction=None, **kwargs):
        """
        Starts the execution of the stencil:

            halo            a tuple defining a 2D halo over the given parameters;
            k_direction     defines the execution direction in 'k' dimension,
                            which might be any of 'forward' or 'backward'.-
        """
        import ctypes

        #
        # make sure all needed parameters were provided
        #
        for p in self.scope.get_parameters ( ):
            if p.name not in kwargs.keys ( ):
                raise ValueError ("Missing parameter '%s'" % p.name)
        #
        # generate one compilation unit for all stencils combined
        #
        if self.backend == 'c++':
            #
            # automatic compilation only if the library is not available
            #
            if Stencil.compiler.lib_obj is None:
                self.resolve (**kwargs)
                #
                # add extra parameters needed to connect linked stencils
                #
                for n,d in self.data_graph.nodes_iter (data=True):
                    succ = self.data_graph.successors (n)
                    if len (succ) == 0:
                        self.scope.add_parameter (n,
                                                  value=d['value'],
                                                  read_only=False)
                    elif len (succ) == 1:
                        self.scope.add_parameter (n,
                                                  value=self.data_graph.node[succ[0]]['value'],
                                                  read_only=True)
                #
                # continue generating the code ...
                #
                self.generate_code ( )
                Stencil.compiler.compile (self)
            #
            # prepare the list of parameters to call the library function
            #
            lib_params = list (self.inspector.domain)

            #
            # extract the buffer pointers from the NumPy arrays
            #
            for p in self.scope.get_parameters ( ):
                if p.name in kwargs.keys ( ):
                    lib_params.append (kwargs[p.name].ctypes.data_as (ctypes.c_void_p))
                else:
                    logging.debug ("Adding linked parameter '%s'" % p.name)
                    linked_param = self.data_graph.node[p.name]['value']
                    if linked_param is None:
                        for s in self.data_graph.successors (p.name):
                            if self.data_graph.node[s]['value'] is not None:
                                linked_param = self.data_graph.node[s]['value']
                    lib_params.append (linked_param.ctypes.data_as (ctypes.c_void_p))
            #
            # call the compiled stencil
            #
            self.compiler.run_native (self, **kwargs)
        elif self.backend == 'python':
            #
            # execution order is from the leafs towards the root
            #
            for st in nx.dfs_postorder_nodes (self.execution_graph,
                                              source=self.get_root ( )):
                st.backend = self.backend
                params     = self._prepare_parameters (st,
                                                       **kwargs)
                st.run (*args,
                        halo=halo,
                        k_direction=k_direction,
                        **params)


    def translate (self):
        """
        Translates this stencil to C++, using the gridtools interface, returning
        a string tuple of rendered files (cpp, make).-
        """
        from gridtools import JinjaEnv

        #
        # instantiate each of the templates and render them
        #
        cpp    = JinjaEnv.get_template ("stencil.cpp")
        make   = JinjaEnv.get_template ("Makefile.cuda")

        params = list (self.scope.get_parameters ( ))
        temps  = list (self.scope.get_temporaries ( ))

        #
        # stencil list and functor dictionaries in execution order
        #
        stencils             = list ( )
        functors             = dict ( )
        independent_functors = dict ( )
        for st in nx.dfs_postorder_nodes (self.execution_graph,
                                          source=self.get_root ( )):
            if st in stencils:
                logging.warning ("Ignoring mutiple instances of stencil '%s' in '%s'"
                                 % (st, self))
            else:
                stencils.append (st)
            functors[st.name]             = list ( )
            independent_functors[st.name] = list ( )
            for f in st.inspector.functors:
                if f in functors[st.name] or f in independent_functors[st.name]:
                    logging.warning ("Ignoring mutiple instances of functor '%s' in stencil '%s'"
                                     % (f, st))
                else:
                    if f.independent:
                        independent_functors[st.name].append (f)
                    else:
                        functors[st.name].append (f)
        #
        # make sure there is at least one non-independent functor in each stencil
        #
        for st in stencils:
            if len (functors[st.name]) == 0:
                functors[st.name]             = [ st.inspector.functors[-1] ]
                independent_functors[st.name] = independent_functors[st.name][:-1]

        return (cpp.render (fun_hdr_file         = self.fun_hdr_file,
                            stencil_name         = self.name.lower ( ),
                            stencils             = stencils,
                            scope                = self.scope,
                            params               = params,
                            temps                = temps,
                            params_temps         = params + temps,
                            functors             = functors,
                            independent_functors = independent_functors),
                make.render (stencil  = self,
                             compiler = self.compiler))

