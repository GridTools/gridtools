# collect test cases
fetch_x86_tests(.)
fetch_mc_tests(.)
fetch_gpu_tests(.)

# collect the cache redcution and conditionals tests.
#those are only enabled when using structured grids
if (STRUCTURED_GRIDS)
    fetch_x86_tests(caches)
    fetch_mc_tests(caches)
    fetch_gpu_tests(caches)
    fetch_x86_tests(reductions)
    fetch_mc_tests(reductions)
    fetch_gpu_tests(reductions)
    fetch_x86_tests(conditionals)
    fetch_gpu_tests(conditionals)
endif()

# collect the grid specific test cases
if (NOT STRUCTURED_GRIDS)
    fetch_x86_tests(icosahedral_grids)
    fetch_mc_tests(icosahedral_grids)
    fetch_gpu_tests(icosahedral_grids)
else()
    fetch_x86_tests(structured_grids)
    fetch_mc_tests(structured_grids)
    fetch_gpu_tests(structured_grids)
    # multi test is a custom test case that tests multiple variations
    foreach( variation CALL PROCEDURES )
        string(TOLOWER ${variation} var_suffix)
        set(ccflags_block "-DFUNCTIONS_${variation} ")
        set(ccflags_naive "-DFUNCTIONS_${variation} -DBACKEND_STRATEGY_NAIVE")
        set(ldflags "${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS}")
        add_custom_x86_test(multi_types_block_${var_suffix} structured_grids/multi_types_test.cpp ${ccflags_block} ${ldflags})
        add_custom_x86_test(multi_types_naive_${var_suffix} structured_grids/multi_types_test.cpp ${ccflags_naive} ${ldflags})
        add_custom_mc_test(multi_types_block_${var_suffix} structured_grids/multi_types_test.cpp ${ccflags_block} ${ldflags})
    endforeach(variation)
endif()
