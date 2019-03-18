.. include:: ../defs.hrst

.. _glossary:

========
Glossary
========

.. glossary:: :sorted:

  Accessor
    A class with an interface to access one element of a data field and its
    neighbors. See section :ref:`stencil_operators`.

  Apply-Method
    Main method of a :term:`Stencil Operator`. Multiple overloads can exist for
    different :term:`Vertical Intervals<Vertical Interval>`. See section
    :ref:`stencil_operators_apply_method`.

  Axis
    A class to generate :term:`Vertical
    Regions<Vertical Region>` of an :term:`Iteration Space`. See section
    :ref:`stencil-composition`.

  Backend
    |GT| provides several backends, allowing to run the same code efficiently
    on different hardware architectures. See section :ref:`backend-selection`.

  Cache
    See :term:`Software-Managed Cache`.

  Computation
    A sequence of :term:`Multi-Stages<Multi-Stage>`
    which is applied on a :term:`Grid` with the :term:`run-method` as an entry-point.
    See section :ref:`composition_of_stencils`.

  Data Store
    An object that manages a logical multidimensional array of values.
    See section :ref:`data-store`.

  Data View
    An object that allows to access and modify the elements of a
    :term:`Data Store` through a tuple of indices. See section
    :ref:`data-view`.

  Execution Model
    Definition of the order of execution of a |GT|
    :term:`Computations<Computation>`. See section :ref:`execution-model`.

  Vertical Execution Order
    A partial order of the vertical axis on an iteration space. See section
    :ref:`execution-model`.

  Extent
    The maximum relative offsets along the coordinate axes at which data is
    accessed around the :term:`Iteration Point`. See section
    :ref:`stencil_operators`.

  GCL
    The |GT| Communication Module. See section :ref:`halo-exchanges`.

  Global Boundary
    Boundary of the :term:`Global Domain` when using distributed computing. See
    section :ref:`distributed-boundary-conditions`.

  Global Domain
    The compute domain that is composed from all domains of the subprocesses
    that participate in a distributed computation. See section
    :ref:`distributed-boundary-conditions`.

  Global Parameter
    A special 0-dimensional :term:`Data Store` for read-only data.
    See section :ref:`global-accessor`.

  Grid
    The grid is the object that defines the :term:`Iteration Space`. See
    section :ref:`defining_iteration_space`.

  Halo
    Additional data points around the main compute data. Used for boundary
    conditions or distributed computations (see :term:`Halo Exchanges<Halo
    Exchange>`). :term:`Halo` information is also stored in the :term:`Storage
    Info` to allow correct data alignment. The size of the :term:`Halo`
    regions is often described by a :term:`Halo Descriptor`. For boundary
    conditions, see sections :ref:`boundary-conditions` and
    :ref:`distributed-boundary-conditions`.

  Halo Descriptor
    An object defining the :term:`Halo` regions of a :term:`Computation`. See
    section :ref:`halo-descriptor`.

  Halo Exchange
    Communication of :term:`Halo` data points between multiple processes. See
    section :ref:`halo-exchanges` and section
    :ref:`distributed-boundary-conditions`.

  Iteration Point
    A 3D tuple of indices. The horizontal indices are often denoted by `i` and
    `j`, while `k` refers to the vertical index.

  Iteration Space
    A set of :term:`Iteration Points<Iteration Point>` on which a stencil is be
    applied. See section :ref:`defining_iteration_space`.

  Layout Map
    A (compile time) sequence of integer values, defining the storage layout of
    a :term:`Data Store`. See section :ref:`storage-module`. Also used to
    define the process layout in an multi-process setup. For this usage, see
    section :ref:`storage-info`.

  Multi-Stage
    A partially-order collection of :term:`Stages<Stage>` with an associated
    :term:`Vertical Execution Order`. See section :ref:`composition_of_stencils`.

  Placeholder
    Placeholders allow compile-time naming of :term:`Stencil Operator`
    arguments. See section :ref:`placeholders`.
    
  Run-Method
    Method to execute a :term:`Computation`, see :ref:`stencil-composition`. 

  Software-Managed Cache
    User-defined caching of fields during a :term:`Multi-Stage`, which has a
    :ref:`cache-policy` and a :ref:`cache-type`.
    See section :ref:`caches`.

  Stage
    A :term:`Stencil Operator` with associated
    :term:`Placeholders<Placeholder>`. See section
    :ref:`composition_of_stencils`.

  Stencil Operator
    Struct or class that defines a stencil operation. See section
    :ref:`stencil_operators`.

  Storage Info
    This concept describes the dimensions, alignment and layout of a
    multidimensional array. See section :ref:`storage-info`.

  Vertical Interval
    A compile-time defined subset of an :term:`Iteration Space`, possibly
    covering only parts of the vertical iteration range. See section
    :ref:`vertical_regions`.

  Vertical Region
    A :term:`Vertical Interval` with associated run-time vertical iteration
    range. See section :ref:`vertical_regions`.
