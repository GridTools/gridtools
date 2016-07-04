Advanced Tutorial
=================

Stages
------

A stencil is composed by one or more *stages*. Each stage defines an operation
to be executed on an array. Stages can be defined in two ways: by directly
writing a for loop calling a ``get_interior_points`` function in the kernel, or
(only for object-oriented stencil) by defining a new method with a name
starting with ``stage_``; the new method must contain a for loop calling a
``get_interior_points`` in order to be completely recognized and processed as a
stage. Stage methods can then be called  in the kernel.

.. note::
    Currently, the only way to create stages in procedural stencils is to use
    for loops in the kernel.

The Game of Life example in the :ref:`tutorial` was a stencil with a single
stage, defined with a for loop in the kernel. Here is another example of a
for-loop stage, with a stencil implementing the discrete Laplace operator:

.. code-block:: python

    class Laplace (MultiStageStencil):
        """
        A Laplacian operator.
        """
        def __init__ (self):
            super ( ).__init__ ( )

        @Stencil.kernel
        def kernel (self, out_data, in_data):
            """
            Stencil's entry point.-
            """
            for p in self.get_interior_points (out_data):
                out_data[p] = -4.0 * in_data[p] + (
                              in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                              in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )

Conversely, here is a stencil implementing Horizontal Diffusion (also called a
fourth-order smoothing filter), showcasing stages defined with functions:

.. code-block:: python

    class HorizontalDiffusion (MultiStageStencil):
        def __init__ (self, domain):
            super ( ).__init__ ( )
            #
            # temporary data fields to share data among the different stages
            #
            self.lap = np.zeros (domain)
            self.fli = np.zeros (domain)
            self.flj = np.zeros (domain)

        def stage_flux_i (self, out_fli, in_lap):
            for p in self.get_interior_points (out_fli,
                                               ghost_cell=[-1,0,-1,0]):
                out_fli[p] = in_lap[p + (1,0,0)] - in_lap[p]

        def stage_flux_j (self, out_flj, in_lap):
            for p in self.get_interior_points (out_flj,
                                               ghost_cell=[-1,0,-1,0]):
                out_flj[p] = in_lap[p + (0,1,0)] - in_lap[p]

        @Stencil.kernel
        def kernel (self, out_data, in_data, in_wgt):
            #
            # Laplace operator
            #
            for p in self.get_interior_points (self.lap,
                                               ghost_cell=[-1,1,-1,1]):
                self.lap[p] = -4.0 * in_data[p] +  (
                              in_data[p + (-1,0,0)] + in_data[p + (1,0,0)] +
                              in_data[p + (0,-1,0)] + in_data[p + (0,1,0)] )
            #
            # Fluxes in the i and j directions
            #
            self.stage_flux_i (out_fli = self.fli,
                               in_lap  = self.lap)
            self.stage_flux_j (out_flj = self.flj,
                               in_lap  = self.lap)

            for p in self.get_interior_points (out_data):
                #
                # Last stage
                #
                out_data[p] = in_wgt[p] * (
                              self.fli[p + (-1,0,0)] - self.fli[p] +
                              self.flj[p + (0,-1,0)] - self.flj[p] )

Stage defined with functions provide for simpler code, because they can be
called multiple times and with different arguments, making the kernel function
easier to write and understand.

Temporary arrays
----------------

The Horizontal Diffusion stencil in the last section also demonstrated another
feature of object-oriented stencils: *temporary arrays*. Any Numpy array defined
in the stencil constructor will be mapped by the GridTools native backend
to a data type specific for temporary (i.e. intermediate) buffers, hence the name.
Temporary arrays are useful to share data among different stages, but
since they have to be defined inside the constructor, they can only be used by
object-oriented stencils.


Data dependency and stage execution path plots
----------------------------------------------

One of the advantages of Gridtools4Py is the possibility to visualize
various characteristics of the stencils the user is working with.
Two functions are currently available:

*   :func:`gridtools.stencil.Stencil.plot_data_dependency`

    Plots the data dependency graph for the stencil: each symbol
    detected in the stencil scope is represented by a colored circle. Every data
    field is connected by an arrow to the data field(s) it depends on (this
    is a directed graph). The color of the circles depends on the symbol kind
    according to the following criteria:

    *   Red: Parameters
    *   Magenta: Aliases
    *   Green: Temporary arrays
    *   Yellow: Constants
    *   Cyan: Local variables
    *   White: Symbol kind could not be determined

    Before asking for the plot it is necessary to run the stencil, to have it
    analyzed by the GridTools4Py's machinery. Note that only the symbols known
    at stencil's scope (i.e., the ones defined in the kernel function) will be
    displayed. Data fields and other variables defined inside stages belong to
    those stages' own scopes and will not be shown in this plot. This function
    is a method of the :class:`Stencil` class, so it must be called as an
    attribute of a stencil object:

    .. code-block:: python

        domain = (64, 64, 32)

        output = np.zeros (domain)
        weight = np.ones  (domain)
        inputs = np.zeros (domain)

        for i in range (domain[0]):
            for j in range (domain[1]):
                for k in range (domain[2]):
                    inputs[i,j,k] = i**5 + j

        hd_stencil = HorizontalDiffusion (domain)
        hd_stencil.set_halo ( (2, 2, 2, 2) )
        hd_stencil.set_k_direction ("forward")

        # Running the stencil is necessary for it to be analyzed by
        # the GridTools4Py's machinery
        hd_stencil.run(out_data = output,
                       in_wgt   = weight,
                       in_data  = inputs)

        # Now we can plot the data dependency graph
        hd_stencil.plot_data_dependency()

    In the case of a procedural stencil, like the Game of Life used in the
    :ref:`tutorial`:

    .. code-block:: python

        game_of_life.plot_data_dependency()

    It is also possible to plot data dependencies from a single stage. In order
    to do this, the stage's data dependency graph and scope have to be explicitly
    passed as arguments to the function:

    .. code-block:: python

        # Plot data dependency graph for the last stage
        hd_stencil.plot_data_dependency(hd_stencil.stages[3].get_data_dependency(),
                                        scope = hd_stencil.stages[3].scope)

        # Plot data dependency graphs for all stages in succession
        for stg in hd_stencil.stages:
            hd_stencil.plot_data_dependency(stg.get_data_dependency(),
                                            scope = stg.scope)

    A legend for nodes colors can be displayed using the appropriate keyword
    argument:

    .. code-block:: python

        hd_stencil.plot_data_dependency(show_legend=True)

*   :func:`gridtools.stencil.Stencil.plot_stage_execution`

    Plots the stage execution graph for the stencil. This directed graph
    illustrates the way stages are arranged in the stencil and the order in
    which they will be executed. As for the data dependency plot, this function
    can be called as a stencil attribute after the stencil has been run at least
    once:

    .. code-block:: python

        hd_stencil.plot_stage_execution()


Vertical Regions
----------------

Creating vertical regions is way of making a stage perform slightly
different operations on different portions of the domain along the vertical
(``k`` index) direction. Vertical regions are defined using Python's slice
syntax on the array argument of ``get_interior_points``.

Here is an example of a stencil implementing slight variations of a Laplacian
operator on different vertical regions:

.. code-block:: python

    class VerticalRegions (MultiStageStencil):
        """
        A stencil using a Laplacian-like operator with different vertical regions
        """
        def __init__ (self):
            super ( ).__init__ ( )


        def stage_laplace0 (self, out_data, in_data):
            for p in self.get_interior_points (out_data[:,:,0:4]):
                out_data[p] = -4.0 * in_data[p] + (
                              in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                              in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


        def stage_laplace1 (self, out_data, in_data):
            for p in self.get_interior_points (out_data[:,:,3:8]):
                out_data[p] = -6.0 * in_data[p] + (
                              in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                              in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


        def stage_laplace2 (self, out_data, in_data):
            for p in self.get_interior_points (out_data[:,:,6:]):
                out_data[p] = -8.0 * in_data[p] + (
                              in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                              in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


        def stage_laplace3 (self, out_data, in_data):
            for p in self.get_interior_points (out_data[:,:,4:8]):
                out_data[p] = -10.0 * in_data[p] + (
                              in_data[p + (1,0,0)]  + in_data[p + (0,1,0)] +
                              in_data[p + (-1,0,0)] + in_data[p + (0,-1,0)] )


        @Stencil.kernel
        def kernel (self, out_data0, out_data1, out_data2, out_data3, in_data):
            self.stage_laplace0 (out_data = out_data0,
                                in_data = in_data)
            self.stage_laplace1 (out_data = out_data1,
                                in_data = in_data)
            self.stage_laplace2 (out_data = out_data2,
                                in_data = in_data)
            self.stage_laplace3 (out_data = out_data3,
                                in_data = in_data)

.. warning::
    The definition of vertical regions has a number of limitations and caveats.
    Please read about them in the dedicated page: :ref:`vr_limitations`
