
Storages
========

Simple use and multidimensional fields
--------------------------------------

Ghost dimensions
----------------

Fields
------

GPU storages
------------

Detailed storage configuration
------------------------------

DSL introduction
================

Syntax for User Operators
-------------------------

### Detailed syntax: , (), dimensions, expressions, offsets of accessors

Parallelization Model: Stages and MultiStages
---------------------------------------------

Preparing arguments: storage placeholders
-----------------------------------------

Vertical regions and vertical boundary conditions
-------------------------------------------------

Use of temporaries
------------------

Composing Stencils
------------------

Irregular Grids Syntax
----------------------

Stencil Functions
=================

Conditional Stencils
====================

Reductions
==========

Details for performance
=======================

Fusion and organization of stages
---------------------------------

Software Managed Caches
-----------------------

Software managed caches are a very important functionality of $\GT$
in order to exploit performance for stencils on computing architectures,
like NVIDIA GPUs, where cache of data is not automatic. A significant
optimization of the stencils using $\GT$ is achieved by means of an
increase of the data locality of the stencil algorithms, for which
detecting data reuse patterns and caching the corresponding fields is
essential for a good performance.

Since GPUs require to explicit express in the code the use of the
different memory hierarchies (i.e. shared memory, registers, texture
cache... ) $\GT$ provides a special syntax to express the data reuse
patterns. The syntax is independent of the type of scratch pad used by
the library to store data and depends only on the access patterns of 
that field by the
stencil methods. The main syntax for caching certain fields of a
`computation` is shown below

~~~~~~~~{.cpp}
auto comp = make_computation< BACKEND >(
    domain,
    grid,
    make_multistage(
        execute< forward >(),
        define_caches(cache< IJ, local >(p_f1(), p_f2())),
        make_stage< lap_function >(p_f1(), p_f2(), p_in()),
        make_stage< lap_function >(p_out(), p_f1(), p_f2())
    )
);
~~~~~~~~

The cache DSL elements are enclosed into a `define_caches`
construct, that accept any number of `cache` constructs. At the same
time, each `cache` construct can specify multiple fields that shared
the same access pattern.

The `cache` construct adheres to the following syntax:

    cache< cache_type, io_policy, [interval] >( p_arg) 

There are multiple type of caches that can be used for different data
reuse pattern situations:

1.  IJ caches: cache fields whose access pattern lies in the IJ-plane

2.  K caches: cache field whose access pattern is restricted to the
    K-plane

3.  bypass: Special cache-type that express null or very little reuse
    within the stencil

Additionally the cache

Alignment: halo of storages
---------------------------

Expandable parameters
---------------------

Halo Updates
============

Boundary Conditions
-------------------

Halo Exchanges
--------------

Data management
===============

Interfacing to other programming languages
==========================================

Benchmarking stencils
=====================

SerialBox and porting your reference application
================================================

[^1]: With the Cuda backend we allocate memory on host and device. In
    the standard use-cases you don’t need to update the data manually,
    but you still have the option to do so.

[^2]: [``]{} is not yet supported.

[^3]: At this point the reader should be able to complete the missing
    parts in the setup.

[^4]: There are other ways to accomplish this behavior. The extra
    computation can be avoided by defining the Laplacian only on the
    interval where we need it; the temporary could be avoided by a bit
    of code duplication, however there is no good reason to do it.
