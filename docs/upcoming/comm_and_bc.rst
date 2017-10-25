=======================================
 Communication and Boundary Conditions
=======================================

----------------------
A recap on boundary conditions
----------------------

The boundary conditions in GridTools (so far), works as follows:

.. code-block:: c++

  boundary_apply< bc_class, predicate >(halos, bc_class(...), predicate(...)).apply(data_stores...);


The number of data stores in the `apply` call depends on `operator()` of boundary condition class. The body of this operator is arbitrary. I will try to categorize the different possible boundary condition types that we may have:

* **Self apply**: The application of the boundary condition to a data store or field depends only on the data store or field itself (or the state of the boundary condition class, which is not modified during the body of `operator() const` (no mutable data members).
    * **Single argument**: `operator()` takes only one argument
    * **Multiple argument**: `operator()` takes two or more arguments
    * **Variadic**: `operator()` has a variadic argument list, and the boundary operation is the same for all the arguments provided, executed one argument at the time
* **Correlated**: The body of `operator()` perform complex operations involving the input data stores or fields and the state of the boundary condition class (in what follows `operator()` is assumed to be `const` anyway)
    * **Copy to one**: `opetator()` has two arguments, one is read-only, the other is written. The convention of which argument is the source is usually not mandated
    * **Copy to many**: `operator()` has more than arguments, one is read-only, the others are written. To compute the value in one data store or field, only the read only one the same data sotre or field are used. The convention of which argument is the source is usually not mandated, but it is usually the first of the last (a variadic version is possible if the first argument is the source)
    * **Many to Many**: `opetator()` has many arguments but no restrictions are in place (it's difficult to imagine a variadic version of this)

-----------------
 A recap on GCL
-----------------

What follows is a snippets

.. code-block:: C++

  typedef gridtools::halo_exchange_dynamic_ut< layoutmap,
    gridtools::layout_map< 0, 1, 2 >,
    triple_t< USE_DOUBLE >::data_type,
    gridtools::MPI_3D_process_grid_t< 3 >,
    arch_type,
    version > pattern_type;
  
  pattern_type he(typename pattern_type::grid_type::period_type(per0, per1, per2), CartComm);
  
  he.template add_halo< 0 >(H1, H1, H1, DIM1 + H1 - 1, DIM1 + 2 * H1);
  he.template add_halo< 1 >(H2, H2, H2, DIM2 + H2 - 1, DIM2 + 2 * H2);
  he.template add_halo< 2 >(H3, H3, H3, DIM3 + H3 - 1, DIM3 + 2 * H3);
  
  he.setup(3);
  
  he.post_receives();
  
  he.pack(vect); // vector of pointers to raw storages
  
  he.do_sends();
  
  he.wait();
  
  he.unpack(vect);

^^^^^^^^^^^^^^^^^^^^^^
 A bottom-up approach
^^^^^^^^^^^^^^^^^^^^^^

I think the best way to approach the problem of designing a communication-and-boundary-condition (CABC) layer in GridTools should be build bottom-up. I list here the requirements that have been expressed so far:

* When doing expandable parameters or data fields, the user may want to apply BCs and perform communications on a sub-set of the data stores collected in these data representations. For this reason an interface for CABC should take just data-stores.
* The user may want to apply different BCs to the same data-store at different times during an executions, so the binding between BCs and data-stores should be done at member-function level, not at class level, in order to remove the need for instantiation of heavy objects like halo-updates.
* The same holds for the data stores to be exchanged: we need to plug the data stores at the last minute before doing the packing/unpacking and boundary apply.
* The halo exchange patterns are quite heavy objects and the lines in the snippet above here, up to the call to `setup`, need to be executed only once to prevent memory leaks. An alternative is to construct and destroy the pattern at every communication, which is not a clean approach and will suffer performance penalties. The information about the pattern must be provided to the CABC set up (that is CABC object construction).
* The halo information can be derived by a `storage_info` class, but there may be cases in which a separate halo information can be provided. For this reason I propose to have a `halo_descriptor` interface that models a simplified `storage_info` in terms of retrieving halo information.
* The `value_type` should be passed as an additional template parameter to the CABC class template.


There are two communication patterns that we can use. First `halo_exchange_ut`, for cases in which the sizes and value types of elements are known, as long as the maximum number of data stores to be updated is bounded. Second, the `halo_exchange_generic`, in which halo and storage sizes are known only at exchange time (the maximum amount of data exchanged must be known before hand for this case, too). The `_generic` one offers more flexibility at the cost of some runtime overhead (not really big in my benchmarks), while the `_ut` is more restricted but faster, and probably also a little more easy to use. In fact, the `_generic` works well with a variadic interface to pass the data stores (pointers) one after the others, since they will have different halo sizes and value types. The `_ut` offers a *vector interface* in which the pointers to the data stores, that (basically) share the same `storage_info` type, can be put in a `std::vector`. The `_generic` also offer the vector interface, but only for when the data stores share the same value type (the GCL accepts pointers to data and separate halo descriptors, so it is not strictly necessary they share the same halo information.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 A proposal for using the `_ut`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A CABC class should contain a specific instantiated `halo_exchange_ut` object, constructed at the beginning. The boundary condition application can be indicated with a bundled object when an operation is requested.

.. code-block
 using cabc_t = CABC<comm_traits> // comm_traits obtained possibly from a prepackaged sets
                                 // of traits containing layouts, data_types, processor grids,
                                 // architecture and packing method
                                 // In addition also the BC predicate
 
 cabc_t cabc(halo_decsriptors, // obtained by grid or storages: array<halo_decsriptor, 3>
            periodicity,      // as in old GCL: Maybe this could be left as last argument ad defaulted
            max_data_stores); // maximum number of data stores used at the same time in this pattern
 
 cabc.exchange(ds0, ds1, ds2); // Exchange data on those data stores
 
 cabc.exchange( bind_bc(bc_class0(...), ds0, ds1), bind_bc(bc_class1(...), ds2) );


Ideally, the pairing of calling `exchange` and `bind_bc` is extracting from the arguments passed to a boundary condition class the data stores that need to be updated. So in case of _correlated_ boundary apply, the read-only data stores should not be updated. Otherwise we could limit `bind_bc` to work on self applying boundary conditions.

An alternative may be something like the following, using the standard way of binding arguments to functions.

.. code-block
 bind_bc(bc_class(...), ds0, ds1, _1).associate(ds_read_only)


.. note::

 It may be that the use cases are much easier than that. If this is the case, the interface could be much easier. Seen next. It may make sense to have this simplified interface for some applications, but I'm not sure if they could be valid in a generic library.

