.. include:: ../defs.hrst

.. _glossary:

=========
Glossary
=========

.. glossary:: :sorted:

  Backend
    |GT| provides several backends, allowing to run the same code efficiently
    on different hardware architectures. See :ref:`backend-selection`.

  Iteration Point
    A 3D tuple of indices. The horizontal indices are often denoted by `i` and
    `j`, while `k` refers to the vertical index.

  Iteration Space
    A set of :term:`Iteration Points<Iteration Point>` on which a stencil is be
    applied. See section :ref:`defining_iteration_space`.

  Execution Model
    Definition of the order of execution of a |GT|
    :term:`Computations<Computation>`. See section :ref:`execution-model`.

  Execution Order
    A partial order of the vertical axis on an iteration space. See section
    :ref:`execution-model`.

  Grid
    The grid is the object that defines the :term:`Iteration Space`. See
    section :ref:`defining_iteration_space`.

  Layout Map
    A (compile time) sequence of integer values, defining the storage layout of
    a :term:`Data Store`. See section :ref:`storage-module`. Also used to
    define the process layout in an multi-process setup. For this usage, see
    section :ref:`storage-info`.

  Storage Info
    This concept describes the dimensions, alignment and layout of a
    multidimensional array. See section :ref:`storage-info`.

  Data Store
    A ``data_store`` object manages a logical multidimensional array of values.
    See section :ref:`data-store`.

  Data View
    A ``data_view`` object allows to access and modify the elements of a
    :term:`Data Store` through tuple of indices. See section
    :ref:`data-view`.

  Extent
    The maximum relative offsets at which data is accessed around the
    :term:`Iteration Point`. See section :ref:`stencil_operators`.

  Accessor
    An object with an interface to access one element of a data field and its
    neighbors. See section :ref:`stencil_operators`.

  Global Accessor
    An object similar to a regular :term:`Accessor`, but allowing access to the
    same (single) read-only data value on the whole :term:`Iteration Space`.
    See section :ref:`global-accessor`.

  Vertical Interval
    A compile-time defined subset of an :term:`Iteration Space`, possibly
    covering only parts of the vertical iteration range. See
    :ref:`vertical-regions`.

  Vertical Region
    A :term:`Vertical Interval` with associated run-time vertical iteration
    range. See :ref:`vertical-regions`.

  Axis
    An object consisting of a collection of disjoint :term:`Vertical
    Regions<Vertical Region>` of an :term:`Iteration Space`. See
    :ref:`stencil-composition`.

  Stencil Operator
    Struct or class that defines a stencil operation. See
    :ref:`stencil_operators`.

  Placeholder
    Placeholders allow compile-time naming of :term:`Stencil Operator`
    arguments. See :ref:`placeholders`.

  Stage
    A :term:`Stencil Operator` with associated
    :term:`Placeholders<Placeholder>`. See :ref:`composition_of_stencils`.

  Multi-Stage Computation
    A partially-order collection of :term:`Stages<Stage>` with an associated
    :term:`Execution Order`. See :ref:`composition_of_stencils`.

  Computation
    A sequence of :term:`Mutli-Stage Computations<Multi-Stage Computation>`
    associated with a :term:`Grid`. This is an executable composition of
    multiple stencils. See :ref:`composition_of_stencils`.

  Halo Descriptor
    An object defining the halo regions of a :term:`Data Store`. See section
    :ref:`halo-descriptor`.

  Boundary Condition
    A functor describing boundary conditions for one or multiple :term:`Data
    Stores<Data Store>`. See section :ref:`boundary-conditions`.

  Do-Method
    Main method of a :term:`Stencil Operator`. See section
    :ref:`stencil_operators_do_method`.

  Elementary Stencil
    Application of a single :term:`Stencil Operator` on an :term:`Iteration
    Space`.

  Software-Managed Cache
    User-defined caching of fields during a :term:`Multi-Stage Computation`.
    See section :ref:`caches`.

  Cache
    See :term:`Software-Managed Caches`.

  Cache Type
    The kind of a :term:`Software-Managed Cache`. See section
    :ref:`cache-type`.

  Cache Policy
    The synchronization policy of a :term:`Software-Managed Cache`. See section
    :ref:`cache-policy`.

  Alignment
    Alignment of the first elements along the contiguous data axis in a :term:`Data
    Store`. See section :ref:`storage-info`.

  GCL
    The |GT| Communication Module. See section :ref:`halo-exchanges`.
