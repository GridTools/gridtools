# -*- coding: utf-8 -*-
import logging

import numpy as np
import networkx as nx

from functools import wraps

from gridtools.symbol   import StencilScope
from gridtools.compiler import StencilCompiler
from gridtools.utils    import Utilities



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
    #
    # a global holder for the used backend
    #
    _backend = "python"
    #
    # a global halo descriptor
    #
    _halo = (0, 0, 0, 0)
    #
    # a global holder for the execution direction in the 'k' (3rd) dimension
    #
    _k_direction = 'forward'


    @staticmethod
    def _interior_points_generator (data_field, ghost_cell, halo, k_direction):
        """
        Returns a generator over the 'data_field'.
        This is an internal method that provides a common foundation for the
        get_interior_points() external interfaces (the Stencil static method and
        the MultiStageStencil bound method). Users should not call this method
        directly. Please use the appropriate get_interior_points() function.

        :param data_field:      a NumPy array;
        :param ghost_cell:      access pattern for the current field, useful for
                                concatenating several stencil stages.-
        :param halo:            a tuple-like object describing the halo for this
                                operation. It is defined as:
                                (halo in negative direction over _i_,
                                 halo in positive direction over _i_,
                                 halo in negative direction over _j_,
                                 halo in positive direction over _j_).-
        :param k_direction:     a string-like object indicating the execution
                                direction in the k dimension. Possible values
                                are 'forward' and 'backward'.
        """
        try:
            if len (data_field.shape) != 3:
                raise ValueError ("Only 3D arrays are supported.")

        except AttributeError:
            raise TypeError ("Calling 'get_interior_points' without a NumPy array")
        else:
            #
            # If data_field is a sliced array, we have to get the offsets in
            # order to properly generate the interior point coordinates.
            # The 'base' property will tell if data_field's memory comes from
            # another object.
            # If so, we use the NumPy Array Interface to get pointers to the
            # first data elements of both data_field and its base array.
            # For detalied information, please see
            # http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.interface.html#__array_interface__
            # Using these pointers with data_field's byte strides along each
            # dimension and some old-fashioned matrix-indexing arithmetic, we
            # can finally obtain the slice offsets.
            #
            i_off, j_off, k_off = (0, 0, 0)
            if data_field.base is not None:
                ptr_offset = (data_field.__array_interface__['data'][0]
                              - data_field.base.__array_interface__['data'][0])
                #
                # Check data ordering on base array, because if data_field is
                # sliced on any of the first 2 dimensions it will not be
                # C contiguous nor F contiguous, as data is not on a single
                # contiguous segment
                #
                if data_field.base.flags['C_CONTIGUOUS']:
                    i_off, mod = divmod (ptr_offset, data_field.strides[0])
                    j_off, mod = divmod (mod, data_field.strides[1])
                    k_off, mod = divmod (mod, data_field.strides[2])
                else:
                    k_off, mod = divmod (ptr_offset, data_field.strides[2])
                    j_off, mod = divmod (mod, data_field.strides[1])
                    i_off, mod = divmod (mod, data_field.strides[0])
            #
            # calculate 'i','j','k' iteration boundaries
            # based on slice offsets, 'halo' and field-access patterns
            #
            i_dim, j_dim, k_dim = data_field.shape

            start_i = i_off + 0     + halo[0] - ghost_cell[0]
            end_i   = i_off + i_dim - halo[1] + ghost_cell[1]
            start_j = j_off + 0     + halo[2] - ghost_cell[2]
            end_j   = j_off + j_dim - halo[3] + ghost_cell[3]

            #
            # calculate 'k' iteration boundaries based 'k_direction'
            #
            if k_direction == 'forward':
                start_k = k_off + 0
                end_k   = k_off + k_dim
                inc_k   = 1
            elif k_direction == 'backward':
                start_k = k_off + k_dim - 1
                end_k   = k_off - 1
                inc_k   = -1
            else:
                logging.warning ("Ignoring unknown K direction '%s'" % k_direction)
            #
            # return the coordinate tuples in the correct order
            #
            for i in range (start_i, end_i):
                for j in range (start_j, end_j):
                    for k in range (start_k, end_k, inc_k):
                        yield InteriorPoint ((i, j, k))


    @staticmethod
    def _validate_halo (halo):
        """
        Check that the provided halo is formatted correctly and contains
        acceptable values

        :param halo:        A tuple-like object
        :return:            True if the input halo is valid
        :raise ValueError:  If the input halo has a number of elements different
                            from 4, or if any of the halo elements is not a
                            non-negative integer.
        """
        if halo is None:
            return False
        if len (halo) == 4:
            if halo[0] >= 0 and halo[2] >= 0:
                if halo[1] >= 0 and halo[3] >= 0:
                    return True
                else:
                    raise ValueError ("Invalid halo %s: definition for the positive halo should be zero or a positive integer" % str (halo))
            else:
                raise ValueError ("Invalid halo %s: definition for the negative halo should be zero or a positive integer" % str (halo))
        else:
            raise ValueError ("Invalid halo %s: it should contain four values" % str (halo))


    @staticmethod
    def _get_for_loop_stage (calling_func_name, parentframe, stencil_obj):
        """
        Tries to detect which stage defined with a for-loop is calling a
        get_interior_points() function inside the stencil kernel

        :param calling_func_name: The name of the function calling get_interior_points
        :param parentframe:       The stack frame object above the stack frame
                                  of the get_interior_points function
        :param stencil_obj:       The stencil object currently running
        :return:                  The stage object relative to the get_interior_points
                                  call
        """
        from gridtools import STENCIL_KERNEL_DECORATOR_LABEL
        calling_func = getattr (stencil_obj, calling_func_name)
        if hasattr (calling_func, STENCIL_KERNEL_DECORATOR_LABEL):
            #
            # Find the line number of the call to get_interior_points()
            # relative to the source code stored in the stencil scope.
            # To do this, we take the line number of the call to
            # get_interior_points in the caller frame, and subtract the line
            # offset between the kernel decorator in the calling stack frame
            # and the decorator in the py_src member of StencilScope.
            #
            stage_line = (parentframe.f_lineno
                          - (parentframe.f_code.co_firstlineno
                             - stencil_obj.scope.kernel_lineno))
            #
            # The resulting line number corresponds to the lineno property
            # of the AST node contained in the Stage we are looking for
            # (because the Stage AST node was generated from
            # StencilScope.py_src)
            #
            for stg in stencil_obj.stages:
                if stg.node.lineno == stage_line:
                    calling_stage = stg
                    break
            return calling_stage
        else:
            #
            # Not being called by a stage function or the kernel.
            # Raise an error.
            #
            raise RuntimeError ("Calling get_interior_points() from outside the stencil kernel without specifying a ghost_cell argument.")


    @staticmethod
    def kernel (kernel_func):
        """
        Decorator to define a given function as the stencil entry point (aka kernel).
        The decorator embeds a runtime check to verify it is called from the
        run() function of the stencil.
        """
        import types
        from gridtools import STENCIL_KERNEL_DECORATOR_LABEL
        #
        # The @wraps decorator is useful to correctly set the __wrapped__ attribute
        # of the user-defined entry point, so that when the stencil is processed
        # by the StencilInspector, the kernel information can be extracted easily
        #
        @wraps (kernel_func)
        def kernel_wrapper (*args, **kwargs):
            #
            # Check that the kernel is being called from its own class' run() method
            #
            if not Utilities.check_kernel_caller (args[0]):
                raise RuntimeError ("Calling kernel function from outside run() function. \
                                    Please use run() to execute the stencil.")
            #
            # Pass 'args' (that contains 'self') to the user-defined kernel only
            # if it was defined with OOP
            #
            if len(kernel_func.__qualname__.split('.')) > 1:
                return kernel_func (*args, **kwargs)
            else:
                return kernel_func (**kwargs)

        setattr (kernel_wrapper,
                 STENCIL_KERNEL_DECORATOR_LABEL,
                 True)

        #
        # If the kernel function is the method of a class, it means the stencil
        # is being defined using OOP, and the wrapper should be returned.
        # If the stencil is being defined with the procedural programming style,
        # return an object extending the MultiStageStencil class.
        #
        if len(kernel_func.__qualname__.split('.')) > 1:
            return kernel_wrapper
        else:
            class UserStencil (MultiStageStencil):
                def __init__ (self):
                    super ( ).__init__ ( )

                def __call__ (self, *args, **kwargs):
                    self.run (*args, **kwargs)

                def get_halo (self):
                    return Stencil.get_halo ( )

                def get_k_direction (self):
                    return Stencil.get_k_direction ( )

                def set_halo (self, halo):
                    raise NotImplementedError ("It is not possible to set an \
                        individual halo for a stencil defined with the \
                        procedural programming style")

                def set_k_direction (self, k_direction):
                    raise NotImplementedError ("It is not possible to set an \
                        individual k direction for a stencil defined with the \
                        procedural programming style")

                @property
                def backend (self):
                    return Stencil.get_backend ( )

                @backend.setter
                def backend (self, value):
                    raise NotImplementedError ("It is not possible to set an \
                        individual backend for a stencil defined with the \
                        procedural programming style")

            user_stencil = UserStencil ( )
            #
            # The wrapper must be set as a bound method in order to be detected
            # by the StencilInspector
            #
            setattr (user_stencil,
                     kernel_func.__name__,
                     types.MethodType (kernel_wrapper, user_stencil) )

            return user_stencil


    @staticmethod
    def get_backend ( ):
        return Stencil._backend


    @staticmethod
    def set_backend (value):
        from gridtools import BACKENDS

        if value in BACKENDS:
            Stencil._backend = value
            logging.debug ("Setting global Stencil backend to %s" % str (Stencil._backend))
            Stencil.compiler.recompile ( )
        else:
            logging.warning ("Ignoring unknown backend '%s'" % value)


    @staticmethod
    def get_interior_points (data_field, ghost_cell=None):
        """
        Returns a generator over the 'data_field' without including the halo.
        Uses global Stencil settings for halo and k direction.

        :param data_field:      a NumPy array;
        :param ghost_cell:      access pattern for the current field, which depends
                                on the following stencil stages.  If None is
                                provided, the function will try to find the stage
                                invoking it, and will obtain the ghost cell
                                pattern from that stage-
        """
        if ghost_cell is None:
            import inspect
            #
            # Find the name of the calling stage function
            #
            stack = inspect.stack()
            if len(stack) < 3:
                raise RuntimeError ("Calling get_interior_points() from "+
                                    "outside the stencil kernel without " +
                                    "specifying a ghost_cell argument.")
            parentframe = stack[1][0]
            calling_func_name = parentframe.f_code.co_name
            #
            # Obtain a reference to the current stencil object by looking
            # into the arguments of the kernel wrapper
            #
            wrapperframe = inspect.stack()[2][0]
            stencil_obj = wrapperframe.f_locals['args'][0]
            #
            # Check if get_interior_points() is being called by the kernel
            # to create a for-loop defined sage
            #
            calling_stage = Stencil._get_for_loop_stage(calling_func_name=calling_func_name,
                                                        parentframe=parentframe,
                                                        stencil_obj=stencil_obj)
            ghost_cell = calling_stage.ghost_cell

        return Stencil._interior_points_generator (data_field,
                                                   ghost_cell=ghost_cell,
                                                   halo=Stencil.get_halo ( ),
                                                   k_direction=Stencil.get_k_direction ( ))


    @staticmethod
    def get_halo ( ):
        return Stencil._halo


    @staticmethod
    def set_halo (halo=(0,0,0,0)):
        """
        Applies the received 'halo' setting, which is defined as

            (halo in negative direction over _i_,
             halo in positive direction over _i_,
             halo in negative direction over _j_,
             halo in positive direction over _j_).-
        """
        if Stencil._validate_halo (halo):
            Stencil._halo = halo
            Stencil.compiler.recompile ( )
            logging.debug ("Setting global Stencil halo to %s" % str (Stencil._halo))


    @staticmethod
    def get_k_direction ( ):
        return Stencil._k_direction


    @staticmethod
    def set_k_direction (direction="forward"):
        """
        Applies the execution order in `k` dimension:

        :param direction:   defines the execution order, which may be any of:
                            forward or backward.-
        """
        from gridtools import K_DIRECTIONS

        if direction is None:
            return

        if direction in K_DIRECTIONS:
            Stencil._k_direction = direction
            Stencil.compiler.recompile ( )
            logging.debug ("Setting global Stencil k_direction to '%s'" % Stencil._k_direction)
        else:
            logging.warning ("Ignoring unknown direction '%s'" % direction)


    def __init__ (self):
        #
        # register this stencil with the compiler and inspector
        #
        self.name             = Stencil.compiler.register (self)
        #
        # name of the stencil entry point defined by the user
        # it will be set by the StencilInspector
        #
        self.entry_point_name = ''
        #
        # the domain dimensions over which this stencil operates
        #
        self.domain           = None
        #
        # symbols gathered after analyzing the stencil code are kept here
        #
        self.scope            = StencilScope ( )
        #
        # Splitters dictionary for this stencil: given a slice index, returns
        # the corresponfding splitter number
        #
        self.splitters        = dict ( )


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


    def _plot_graph (self, G, axes=None, outfile=None, **kwargs):
        """
        Renders graph 'G' using 'matplotlib'.-

        :param G:        The graph to plot
        :param axes:     A Matplotlib Axes object on which the graph will be plotted.
                         If no object is specified, a new figure will be created.
        :param **kwargs: Optional keyword arguments that will be passed to
                         networkx.draw_networkx()
        :return:
        """
        from gridtools import plt

        pos = nx.spring_layout (G)

        if axes is None:
            #
            # Create a new figure to host the plot
            #
            fig, axes = plt.subplots (1, 1)

        nx.draw_networkx (G,
                          pos=pos,
                          ax=axes,
                          node_size=1500,
                          **kwargs)
        if outfile:
            plt.savefig(outfile)


    def build_data_dependency (self):
        """
        Build a data dependency graph among this stencil's data fields.

        Stencil scope data dependency is obtained by aggregating each stage
        dependencies and resolving aliases.
        This method should be called after Stencil.generate_code(), when all
        data dependencies are known at stage scope.

        Dependencies involving stage local variables (which are not known at
        stencil scope) are processed by removing the local symbol node and
        creating edges between all predecessors and successors of the node.
        Be aware that this may generate false dependencies when the same local
        variable is reassigned and involved in the computation of two logically
        unrelated data fields!!
        :return:
        """
        for stg in self.stages:
            #
            # Create a copy of stage data dependency to avoid altering the
            # original
            #
            stgdd = nx.DiGraph (stg.get_data_dependency ( ))
            for node in stg.get_data_dependency ( ).nodes_iter ( ):
                new_edges = []
                if stg.scope.is_local (node):
                    #
                    # If node represents a local symbol, create direct edges
                    # between all its predecessors and all its successors and
                    # remove the node
                    #
                    for pred in stgdd.predecessors (node):
                        for succ in stgdd.successors (node):
                            new_edges.append ((pred, succ))
                    stgdd.remove_node (node)
                    stgdd.add_edges_from (new_edges)
            #
            # Add dependencies without local symbols to the stencil's scope
            #
            self.scope.add_dependencies (stgdd.edges ( ))


    def generate_code (self):
        """
        Generates C++ code for this stencil
        :raise Exception: in case of an error during the code-generation process
        :return:
        """
        #
        # generate the code of *all* stages in this stencil
        #
        for stg in self.stages:
            stg.generate_code ( )


    def generate_splitters (self):
        """
        Generates splitters data for this stencil from slicing information
        contained in each stage's vertical regions

        :return:
        """
        #
        # Gather slice indexes from all vertical regions
        #
        slice_indexes = list ( )
        for stg in self.stages:
            for vr in stg.find_slice_indexes (self.scope):
                slice_indexes.append (vr[0])
                slice_indexes.append (vr[1])
        #
        # Remove duplicates and sort slice indexes list
        #
        slice_indexes = sorted (set (slice_indexes))
        #
        # Populate splitters dictionary
        #
        for i, idx in enumerate (slice_indexes):
            self.splitters[idx] = i
        #
        # Save splitter data in each vertical region
        #
        for stg in self.stages:
            stg.set_splitters (self.splitters)


    def get_data_dependency (self):
        """
        Return the data dependency graph for this stencil's scope

        :return:
        """
        return self.scope.data_dependency


    def get_stage (self, name):
        """
        Returns the stage first stage object found containing the 'name'
        argument string in its name

        :raise KeyError: If a matching stage name could not be found
        """
        ret_val = None
        for stg in self.stages:
            if name in stg.name:
                ret_val = stg
                break
        else:
            raise KeyError("No stage found containing '%s' in its name in stencil %s"
                            % (name, self.name))
        return ret_val


    def identify_IO_stages (self):
        """
        Tries to identify input and output data fields for every stage of the
        stencil.

        This method should be called after Stencil.generate_code(), because
        during code generation all data dependencies and aliases are resolved
        in all stages' scopes.
        :return:
        """
        for stg in self.stages:
            stg.identify_IO_fields ( )


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


    def plot_data_dependency (self, graph=None, scope=None, show_legend=False,
                              outfile=None):
        """
        Renders a data-depencency graph using 'matplotlib'.
        Graph nodes are colored based on their kind according to the following
        rule:
        * Red: Parameters
        * Magenta: Aliases
        * Green: Temporaries
        * Yellow: Constants
        * Cyan: Locals
        * White: Symbol kind could not be determined

        :param graph:   the graph to render; it renders this stencil's data
                        dependency graph if None given
        :param scope:   the scope in which to look for symbols. If None, the
                        the stencil's scope will be used
        :param show_legend: boolean flag to display the legend, explaining the
                            node colors
        :param outfile: the name of the file in which to save the plot
        :return:
        """
        if graph is None:
            graph = self.get_data_dependency ( )
        if scope is None:
            scope = self.scope
        #
        # Assign node colors depending on symbol kind
        #
        node_color = ''
        for n in graph.nodes():
            if scope.is_parameter(n):
                node_color += 'r'
            elif scope.is_alias(n):
                node_color += 'm'
            elif scope.is_temporary(n):
                node_color += 'g'
            elif scope.is_constant(n):
                node_color += 'y'
            elif scope.is_local(n):
                node_color += 'c'
            else:
                node_color += 'w'
        #
        # Display a legend with node colors
        #
        if show_legend:
            from gridtools import plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots (1, 1)

            par_patch = mpatches.Patch(facecolor='r', edgecolor='k', label='Parameter')
            ali_patch = mpatches.Patch(facecolor='m', edgecolor='k', label='Alias')
            tmp_patch = mpatches.Patch(facecolor='g', edgecolor='k', label='Temporary')
            con_patch = mpatches.Patch(facecolor='y', edgecolor='k', label='Constant')
            loc_patch = mpatches.Patch(facecolor='c', edgecolor='k', label='Local')
            unk_patch = mpatches.Patch(facecolor='w', edgecolor='k', label='Unknown')

            ax.legend(handles=[par_patch,ali_patch,tmp_patch,
                               con_patch,loc_patch,unk_patch])
        else:
            ax = None

        self._plot_graph (graph, axes=ax, node_color=node_color)


    def plot_stage_execution (self):
        """
        Renders a stage execution graph using 'matplotlib'
        :return:
        """
        graph = self.scope.stage_execution
        self._plot_graph (graph)


    @property
    def stages (self):
        return nx.topological_sort (self.scope.stage_execution)



class MultiStageStencil (Stencil):
    """
    A stencil involving one or more stages. The stages can be implemented as
    functions or for-loops.
    All user-defined stencils should inherit from this class.-
    """
    def __init__ (self):
        super ( ).__init__ ( )
        #
        # defines the way to execute the stencil
        #
        self._backend         = None
        #
        # a halo descriptor
        #
        self._halo            = None
        #
        # define the execution order in 'k' dimension
        #
        self._k_direction     = None


    def get_backend (self):
        """
        Return the execution backend for this stencil
        If a specific backend was not set for this stencil, use global Stencil
        backend
        """
        if self._backend:
            return self._backend
        else:
            return Stencil.get_backend ( )


    def set_backend (self, value=None):
        """
        Set the execution backend for this stencil
        """
        from gridtools import BACKENDS

        #
        # If no argument is provided, reset the object backend, so that
        # global Stencil backend will be used instead
        #
        if value is None:
            self._backend = None
            Stencil.compiler.recompile ( )
            logging.debug ("backend for stencil '%s' has been reset" %
                           self.name)
            return

        if value in BACKENDS:
            self._backend = value
            Stencil.compiler.recompile ( )
            logging.debug ("Setting backend for stencil '%s' to '%s'" %
                            (self.name, self._backend) )
        else:
            logging.warning ("Ignoring unknown backend '%s'" % value)


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


    def get_interior_points (self, data_field, ghost_cell=None):
        """
        Returns a generator over the 'data_field' without including the halo.
        Uses stencil-specific values for halo and k direction, if they have
        been set. Otherwise, uses global Stencil settings.

        :param data_field:      a NumPy array;
        :param ghost_cell:      access pattern for the current field. If None is
                                provided, the function will try to find the stage
                                invoking it, and will obtain the ghost cell
                                pattern from that stage
        """
        if ghost_cell is None:
            import inspect
            #
            # Find the name of the calling stage function
            #
            stack = inspect.stack()
            parentframe = stack[1][0]
            calling_func_name = parentframe.f_code.co_name
            if "stage_" in calling_func_name:
                #
                # get_interior_points() is being called by a function defined
                # stage. Get the ghost cell pattern from the stage object.
                #
                calling_stage = self.get_stage (calling_func_name)
            else:
                #
                # Check if get_interior_points() is being called by the kernel
                # to create a for-loop defined sage
                #
                calling_stage = Stencil._get_for_loop_stage(calling_func_name=calling_func_name,
                                                            parentframe=parentframe,
                                                            stencil_obj=self)
            ghost_cell = calling_stage.ghost_cell

        return Stencil._interior_points_generator(data_field,
                                                  ghost_cell=ghost_cell,
                                                  halo=self.get_halo ( ),
                                                  k_direction=self.get_k_direction ( ))


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
            # Check the minimum halo has been given
            #
            for idx in range (len (self.scope.minimum_halo)):
                if self.scope.minimum_halo[idx] - self.get_halo ( ) [idx] > 0:
                    raise ValueError ("The halo should be at least %s" %
                                      self.scope.minimum_halo)
            #
            # If a specific backend has been set for this stencil, use that.
            # Otherwise, use the global Stencil backend
            #
            backend = self.get_backend ( )
            #
            # run the selected backend version
            #
            logging.info ("Executing '%s' in %s mode ..." % (self.name,
                                                             backend.upper ( )))
            if backend == 'c++' or backend == 'cuda':
                Stencil.compiler.run_native (self, **kwargs)
            elif backend == 'python':
                getattr (self, self.entry_point_name) (**kwargs)
            else:
                raise ValueError ("Unknown backend '%s' set for stencil '%s'" %
                                  (backend, self.name) )


    def get_halo (self):
        """
        Return the halo for this stencil
        If a specific halo was not set for this stencil, use global Stencil halo
        """
        if self._halo:
            return self._halo
        else:
            return Stencil.get_halo ( )


    def set_halo (self, halo=None):
        """
        Applies the received 'halo' setting, which is defined as

            (halo in negative direction over _i_,
             halo in positive direction over _i_,
             halo in negative direction over _j_,
             halo in positive direction over _j_).-
        """
        #
        # If no argument is provided reset the object halo, so that global
        # Stencil halo will be used instead
        #
        if halo is None:
            self._halo = None
            Stencil.compiler.recompile ( )
            logging.debug ("Halo for stencil '%s' has been reset" % self.name)
            return

        if Stencil._validate_halo (halo) :
            self._halo = halo
            Stencil.compiler.recompile ( )
            logging.debug ("Setting halo for stencil '%s' to %s" %
                           (self.name, str (self._halo)) )


    def get_k_direction (self):
        """
        Return the k direction for this stencil
        If a specific direction was not set for this stencil, use global Stencil
        k direction
        """
        if self._k_direction:
            return self._k_direction
        else:
            return Stencil.get_k_direction ( )


    def set_k_direction (self, direction=None):
        """
        Applies the execution order in `k` dimension:

        :param direction:   defines the execution order, which may be any of:
                            forward or backward.-
        """
        from gridtools import K_DIRECTIONS

        #
        # If no argument is provided, reset the object direction, so that
        # global Stencil directiion will be used instead
        #
        if direction is None:
            self._k_direction = None
            Stencil.compiler.recompile ( )
            logging.debug ("k_direction for stencil '%s' has been reset" %
                           self.name)
            return

        if direction in K_DIRECTIONS:
            self._k_direction = direction
            Stencil.compiler.recompile ( )
            logging.debug ("Setting k_direction for stencil '%s' to '%s'" %
                            (self.name, self._k_direction) )
        else:
            logging.warning ("Ignoring unknown direction '%s'" % direction)



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
        from os        import path, makedirs
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

        stage_src       = ''
        self.lib_file     = 'lib%s' % self.name.lower ( )
        self.cpp_file     = '%s.cpp' % self.name
        self.make_file    = 'Makefile'
        self.stg_hdr_file = '%sStages.h' % self.name

        logging.info ("Generating C++ code in '%s'" % self.src_dir)
        #
        # generate the code for *all* stages within the combined stencil
        #
        for st in nx.dfs_postorder_nodes (self.execution_graph,
                                          source=self.get_root ( )):
            for stage in st.inspector.stages:
                stage.generate_code (st.inspector.src)
                st.scope.add_dependencies (stage.get_data_dependency ( ).edges ( ))
            stg_src, _, _  = Stencil.compiler.translate (st)
            stage_src   += stg_src

        with open (path.join (self.src_dir, self.stg_hdr_file), 'w') as stg_hdl:
            stages = JinjaEnv.get_template ("stages.h")
            stg_hdl.write (stages.render (stage_src=stage_src))
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


    def plot_data_graph (self, outfile=None):
        """
        Renders the data graph using 'matplotlib'.-
        """
        self._plot_graph (self.data_graph, outfile)


    def plot_execution_graph (self, outfile=None):
        """
        Renders the execution graph using 'matplotlib'.-
        """
        self._plot_graph (self.execution_graph, outfile)


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
        # stencil list and stage dictionaries in execution order
        #
        stencils             = list ( )
        stages             = dict ( )
        independent_stages = dict ( )
        for st in nx.dfs_postorder_nodes (self.execution_graph,
                                          source=self.get_root ( )):
            if st in stencils:
                logging.warning ("Ignoring mutiple instances of stencil '%s' in '%s'"
                                 % (st, self))
            else:
                stencils.append (st)
            stages[st.name]             = list ( )
            independent_stages[st.name] = list ( )
            for f in st.inspector.stages:
                if f in stages[st.name] or f in independent_stages[st.name]:
                    logging.warning ("Ignoring mutiple instances of stage '%s' in stencil '%s'"
                                     % (f, st))
                else:
                    if f.independent:
                        independent_stages[st.name].append (f)
                    else:
                        stages[st.name].append (f)
        #
        # make sure there is at least one non-independent stage in each stencil
        #
        for st in stencils:
            if len (stages[st.name]) == 0:
                stages[st.name]             = [ st.inspector.stages[-1] ]
                independent_stages[st.name] = independent_stages[st.name][:-1]

        return (cpp.render (stg_hdr_file         = self.stg_hdr_file,
                            stencil_name         = self.name.lower ( ),
                            stencils             = stencils,
                            scope                = self.scope,
                            params               = params,
                            temps                = temps,
                            params_temps         = params + temps,
                            stages               = stages,
                            independent_stages   = independent_stages),
                make.render (stencil  = self,
                             compiler = self.compiler))
