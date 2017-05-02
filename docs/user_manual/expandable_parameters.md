#### Expandable Parameters

Expandable parameters implement a "single stencil multiple storages" pattern.
They are useful when we have a vector of storages which have the same
storage info, and we want to perform the same operation with all of them
(a typical situation when implementing e.g. time differentiation schemes).
Normally this could be achieved by creating a loop and running multiple computation,
but this solution would be inefficient. Another possibility would be to create a storage_list,
and use the [Vector Accessor](vector accessors) inside the Do method to manually
unroll the loop. This second option is tedious, does not allow generic vector length, and it
forces the size of the vector to be compile-time known. The expandable parameters API
solves this problem elegantly, with a minimal code overhead.

The implementation requires the user to specify an integer ```expand_factor```. The vector of
storages is then partitioned into chuncks of ```expand_factor``` size (with a remainder). Each
chunck is unrolled whithin a computation, and for each chunck a different computation is
instantiated. The remainder elements are then processed one by one.

The user must collect the storage pointers in an ```std::vector```
```c++
        using exp_param_t = std::vector< pointer< storage_t > >;
        exp_param_t list_ = {
            &storage1, &storage2, &storage3, &storage4, &storage5, &storage6, &storage7, &storage8};
```
This ```std::vector``` is then used as a storage type ```exp_param_t``` with no differences with respect to
the regular storages.

The only differences with respect to the case without expandable parameters are:
- an ```expand_factor``` has to be passed to the make_computation, defining the size of the
chuncks of expandable parameters we want to unroll in each computation.
- a [Vector Accessor](vector accessor) has to be used instead of a regular one in the Do method

All the rest is managed by $\GT$, so that the user is not exposed to the complexity of the
unrolling, he can reuse the code when the expand_factor changes, and he can resize dynamically the expandable
paramenters vector, for instance by adding or removing elements.
