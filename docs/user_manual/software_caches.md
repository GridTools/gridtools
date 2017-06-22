# Software Managed Caches

Software managed caches are provide significant optimization since
they provide increased data locality for multi-stage stencils in
architectures with user-managed address spaces with different
performance tradeoffs. For instance, they allow the stencil algorithms
to exploit texture memory, shared memory, and register files on GPUs
in cases in which a compiler could not provide such optimizations.

The user is responsible for
detecting data reuse patterns and caching the corresponding fields in order to
fully utilize the hardware resources 
essential for a good performance.
The syntax provided by $\GT$ is independent of the type of hardware resources used by
the library to store data and depends only on the access patterns of 
the fields by the
stencil methods. An example of the syntax for caching certain fields of a
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
                                                      is specified a cache will be used. Using too many caches
                                                      may cause over-use of hardware resources that may lead
                                                      to decrease in performance.
---------------------------------------------------   --------------------------------------------------------


The `cache` construct adheres to the following syntax:

    cache< cache_type, io_policy, [interval] >( p_args... ) 

We now describe the details of each element of the cache constructs.

### Cache Type

`cache_type` depends on the reuse pattern of the data fields. It's
value can be one of the following (where we indicate the basic mean of implementation on the GPUs, so that the user can understand the amount of resources involved):

1.  `IJ_caches`: cache data fields whose access pattern lies in the IJ-plane (Implemented using the shared memory

2.  `K_caches`: cache data field whose access pattern is restricted to the
    K-direction (implemented using the register file of GPUs)

3.  `IJK_caches`: for data fields that are accessed in a three dimensions (not fully supported yet)

4.  `Pointr_caches`: for data fields accessed multiple times in the point of evaluation only (not fully supported yet)

5.  `bypass`: Special cache-type that express null or very little reuse
    within the stencil. This can be specified to disable the default use of texture memory for read only data fields that the library would use. 

Additionally the cache

### Cache policy

`cache_policy` specify what is the relation between the data in the cache and the data in the data fields. The possible values are:

 1. `fetch`: Before starting executing the stencil operators the data
 to be cached has to be fetched from the corresponding data fields
 into the corresponding cache

 2. `flush`: After the execution of the stencil operators the data in
 the cache has to be written back into the data fields.

 3. `fetch_and_flush`: The combination of `fetch` and `flush`

 4. `local`: The data will be produced by the stencil opertators and
 consumed by other operators of the multi-stage stencil.

### Interval

Not sure what to write here


### p_args

The `p_args...` indicate a list of placeholders corresponding to the
data fields for which the specified caching type and policy has been
requested.

