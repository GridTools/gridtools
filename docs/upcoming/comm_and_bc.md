## Communication and Boundary Conditions

I describe here my desiderata and how we can approximate it.

### Runtime registration
Version using runtime registration (using -assumed unique- strings from placeholders)

```cpp
int test() {

    comm_and_bc<storage_type, architecture> c_b(periodicity(true, false), max_fields);

    // x, y, and z are computations

    auto x = make_computation(...
        make_multistage(
            make_stage<f>(a,b,c) // `a` and `b` are inputs
        ));
        
    auto y = make_computation(...
        make_multistage(
            make_stage<f>(a,b,d,e) // `a` and `b` are inputs
        ));

    auto z = make_computation(...
        make_multistage(
OA            make_stage<f>(c,d,f) // `c` and `d` are inputs, `f` is output
        ));

    c_b.register(aggregator); // Registering the placeholders
                              // The must share the same storage_info type
                              // Inside there is a unordered_map of string/storage pointers;
                              // The aggregator also need the placeholders values to be
                              // stored, but the aggregator does not need the unordered_map.
                              // The registration extract only the pairs <string, storage>
                              // from the ones of the given storage_info type. In this way we
                              // can register multiple aggregators and we won't need a global
                              // aggregator with huge type list.
    x.run();

    c_b.equeue(x); // Take the output fields of `x` and stage them for halo-update
                   // In this case the field `c`
    
    y.run();

    c_b.equeue(y); // Take the output fields of `y` and stage them for halo-update
                   // Here we are assuming there are no data dependencies between `x` and `y`
                   // They are `d` and `e`

    c_b.update(z); // Take the input fields of `z` that are listed in the queue of the
                   // update queue of `c_b` and perform halo_exchanges for that.
                   // Can be split in start_update, and wait_updated, etc.
                   // They are `c` and `d`, but not necessarily `e`

    z.run(); // Now `z` can run since it's input fields are updated

}
```
