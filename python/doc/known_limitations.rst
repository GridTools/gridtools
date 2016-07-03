Known limitations
=================

The Python-based DSL provided by Gridtools4Py has several limitations.
Some of them are by design, while others are due to features that still have
to be implemented or improved.

General limitations
-------------------

*   Stencil kernel and stage functions must return ``None``.
*   The following Python comparison operators are not supported inside stages:
    :keyword:`is`, :keyword:`in`, :keyword:`is not`, :keyword:`not in`.
*   Definition of functions and classes inside stencil kernels or stages is not
    supported.
*   Calling non-stage functions inside kernels is not supported.
*   Calling functions inside stages is not supported.
*   When creating an object oriented stencil and explicitly reimplementing the
    constructor, the parent constructor must be called before doing anything else.
*   The kernel function of an object-oriented stencil cannot be called directly.
    To execute the stencil the :func:`gridtools.stencil.MultiStageStencil.run`
    method must be called.
*   All arguments to :func:`gridtools.stencil.MultiStageStencil.run` must
    be passed as keyword arguments. Passing any positional arguments will raise
    a :class:`ValueError` exception.
*   When calling stage functions from the stencil kernel, all arguments must be
    passed as keyword arguments. Positional arguments will be ignored and will
    very likely result in errors when analyzing or running the stencil.
*   for-loops using ``get_interior_points`` (either the :class:`gridtools.stencil.Stencil`
    static function or the :class:`gridtools.stencil.MultiStageStencil` bound
    method) must use ``p`` as the target (the
    variable the loop assigns to). For example:

    .. code-block:: python

        for p in Stencil.get_interior_points(data_field)

    .. note::
        This limitation will be removed in a future release.

*   The syntax for defining array index offsets inside for-loops using
    ``get_interior_points`` must be ``data[p + (tuple)]``. For example:

    .. code-block:: python

      for p in self.get_interior_points (out_data):
          out_data[p] = out_data[p + (1,-1,0)]

    Only the ``+`` operator is supported for creating index offsets.
    Offsets must be specified using a tuple (like shown above), and not with
    single coordinates (eg. ``out_data[i+1, j-1, k]`` is not supported).

    .. note::
        This limitation will be removed in a future release.

*   Only double precision floating point numbers are supported as values for local
    variables defined within a stage
*   A data field cannot be used to assign a value to itself inside a stencil,
    either directly or indirectly. In other words, a data field cannot depend on
    itself.
    More details at https://github.com/eth-cscs/gridtools/wiki/Data-Dependencies-Analysis-in-GridTools


.. _vr_limitations:

Vertical regions limitations
----------------------------

There are several constraints compared to regular Python slicing:

*   Vertical region slicing must have all 3 dimensions (eg. ``data[:,:]`` is not allowed)
*   Using single indexes in vertical region slicing (eg. ``data[:,:,5]``) is not allowed
*   Using partial slices in `i` or `j` directions (eg. ``data[5:,:10,:]``) is not allowed
*   Using negative indexes in the `k` direction (eg. ``data[:,:,:-2]``) is not allowed
*   Using slice steps in any direction (eg. ``data[:,:,5:10:2]``) is not allowed

.. note::
    Some of these may be removed in future releases.

In addition, there are the following conceptual limitations:

*   Vertical regions inside the same stage cannot overlap (this is also a
    constraint enforced by GridTools).
*   The use of slicing to define vertical regions is possible only inside stages
    defined with a function: vertical regions are defined with the same syntax used
    for creating unnamed stages inside the stencil kernel function
    (for-loop + ``get_interior_points`` call). This would make impossible to
    distinguish between an unnamed stage and a vertical region subordinate to a
    stage.

Finally, the current implementation still has these limitations:

*   Slice indexes can only be literal numbers:

    ``data[:,:,12:25]`` OK

    ``data[:,:,bottom:top]`` NO

    ``data[:,:,:domain[2]-1]`` NO

*   Only a single vertical region can be defined inside a stage
