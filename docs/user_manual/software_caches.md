# Software Managed Caches

Software managed caches are syntax elements that are used
to describe data reuse pattern of the stencil computations. 
They are an essential functionality of the GridTools in order
to deliver an efficient implementation of memory bound codes, 
since the library uses
this information to allocate cached fields in a fast on-chip
scratch-pad memory.

In computing architectures like NVIDIA GPUs, where the use of 
the different on-chip memory hierarchy must be explicitly 
declared using the CUDA programming model, the use of software managed 
caches of GridTools increases the data locality of stencil algorithms 
and provides a significant performance speedup. 

While the library is capable of exploiting several on-chip memory layers
(like texture cache, const cache, shared memory, and registers of NVIDIA GPUs) 
the GridTools language is abstracting these underlying memory layers and 
exposes syntax elements that are computing architecture agnostic.   

Therefore the software managed cache syntax should be used by the 
user to describe *only* data reuse patterns, and not type of 
on-chip memory that should be exploited (which is a decision delegated to 
the computing architecture backend of the library).
  
An example of the syntax for caching certain fields of a
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

The cache DSL elements are enclosed into a `define_caches` construct,
that accept any number of `cache` constructs. At the same time, each
`cache` construct can specify multiple fields that shared the same
access pattern.


---------------------------------------------------   --------------------------------------------------------
![Tip](figures/hint.gif){ width=20px height=20px }
                                                      It is important to note that the `cache` specifications
                                                      are prescribing the behavior of the library: if a cache
                                                      is specified, a cache will be used. In the rare case of
                                                      using too many caches a decrease in performance might be
                                                      observed due to saturation of available resources
---------------------------------------------------   --------------------------------------------------------


The `cache` construct adheres to the following syntax:

    cache< cache_type, io_policy, [interval] >( p_args... ) 

Full examples on cache usages can be found in the source code 
[examples/interface1.hpp](https://github.com/GridTools/gridtools/blob/master/examples/interface1.hpp) 
and
[examples/vertical_advection_dycore](https://github.com/GridTools/gridtools/blob/master/examples/vertical_advection_dycore.hpp)

We now describe the details of each element of the cache constructs.

### Cache Type

`cache_type` describes the type of access pattern present in our stencil for the field being cached. It's
value can be one of the following (where we indicate the basic mean of implementation on the GPUs, so that the user can understand the amount of resources involved):

1.  `cache_type::IJ`: cache data fields whose access pattern lies in the IJ-plane, i.e. only offsets of the type `i+-X` or `j+-Y` are allowed. 
(the GPU backend will cached these fields in shared memory)

2.  `cache_type::K`: cache data field whose access pattern is restricted to the
    K-direction, i.e. only offsets of the type `k+-Z` (the GPU backend will cached these fields in the register file of GPUs)

3.  `cache_type::IJK`: for data fields that are accessed in a three dimensions (not fully supported yet)

4.  `bypass`: Special cache-type that express null or very little reuse
    within the stencil. This can be specified to disable the default use of texture memory for read only data fields that the library would use.
    This can be useful in case that type of GPU cache is saturated due to presence of many read only fields in the stencil computation. 

An error in the specification of the `cache_type`, for example using `cache_type::IJ` for a fields that is accessed with k offsets will lead to compile time 
[protection errors](#syntax-compile-time-protections).

### Cache policy

`cache_policy` specifies a synchronization policy between the data in the cache and the data in main memory. A scratch-pad can be used 
in order to allocate temporary computations that do not require data persistency accross multiple stencils. However often the data that is
being cached is already present in main memory fields. In this case, the software managed caches of GridTools gives the possibility 
to specify a cache policy that allows to synchronize the main memory with the cached field. 
The possible values are:

 1. `fill`: fill the scratch-pad buffer with data from main memory field before use.

 2. `flush`: After the execution of the stencil operators the data in
 the cache is written back into the main memory fields.

 3. `fill_and_flush`: The combination of `fill` and `flush`

 4. `local`: The scratch-pad data is not persistent and only available with the scope of a multi-stage.

 5. `bpfill`: Stands for begin-point-flush. This type of `cache_policy` is only valid for cache types with a `k` component. 
Only the head (i.e. positive offsets for forward loop direction or negative offset for backward loop direction) of a kcache buffer is filled 
from main memory at the beginning of the vertical loop. 
This policy can be used for iterative solvers that require only
an initial seed of the data (few vertical k-levels). 
 
 6. `epflush`: Stands for end-point-flush. This type of `cache_policy` is only valid for cache types with a `k` component. 
 Only the tail (i.e. negative offsets for forward loop direction or positive offset for backward loop direction) of a kcache buffer is flushed to 
main memory at the end of the vertical loop. This policy can be used to provide persistency of the field 
 for another  multi-stage that contains a solver that requires only an initial seed of the field (few vertical k-levels) 
 
 
 The following figure graphically depicts an example of all the ordered operations that are executed when a `fill_and_flush`
  cache is used in a forward vertical loop. 
 
 ![Representation of an implementation for a `cache_type::K` that is used within a 
 stencil with extent `<-2,1>` in the vertical dimension and implemented as a ring-buffer with 4 levels (in order to allocate all possible offsetted accesses). The three operations 
 are triggered automatically by the library for a `fill_and_flush` cache when the vertical loop transition from level 9 to level 10.    ](figures/kcache_ex.png){width="0.1\columnwidth"}

### Interval

The interval is a [vertical interval](#vertical-regions-and-vertical-boundary-conditions) that specifies the region on which the scratch-pad of the cache
will be synchronized with main memory, according to the [cache policy](#cache-policy)


### p_args

The `p_args...` indicate a list of placeholders corresponding to the
data fields for which the specified caching type and policy has been
requested.

