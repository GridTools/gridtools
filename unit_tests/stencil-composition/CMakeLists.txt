# collect test cases
fetch_x86_tests(.)
fetch_gpu_tests(.)

# collect the grid specific test cases
if (STRUCTURED_GRIDS)
    fetch_mc_tests(.)

    # collect the cache redcution and conditionals tests.
    #those are only enabled when using structured grids
    fetch_x86_tests(caches)
    fetch_mc_tests(caches)
    fetch_gpu_tests(caches)
    fetch_x86_tests(reductions)
    fetch_mc_tests(reductions)
    fetch_gpu_tests(reductions)
    fetch_x86_tests(conditionals)
    fetch_gpu_tests(conditionals)

    fetch_x86_tests(structured_grids)
    fetch_mc_tests(structured_grids)
    fetch_gpu_tests(structured_grids)
    # multi test is a custom test case that tests multiple variations
    foreach( variation CALL PROCEDURES )
        string(TOLOWER ${variation} var_suffix)
        set(ccflags_block -DFUNCTIONS_${variation} )
        set(ccflags_naive -DFUNCTIONS_${variation} -DBACKEND_STRATEGY_NAIVE)
        add_custom_x86_test(TARGET multi_types_block_${var_suffix} SOURCES structured_grids/multi_types_test.cpp ADDITIONAL_FLAGS ${ccflags_block} )
        add_custom_x86_test(TARGET multi_types_naive_${var_suffix} SOURCES structured_grids/multi_types_test.cpp ADDITIONAL_FLAGS ${ccflags_naive} )
        add_custom_mc_test(TARGET multi_types_block_${var_suffix} SOURCES structured_grids/multi_types_test.cpp ADDITIONAL_FLAGS ${ccflags_block} )
    endforeach(variation)

else()
    fetch_x86_tests(icosahedral_grids)
    fetch_gpu_tests(icosahedral_grids)
endif()
