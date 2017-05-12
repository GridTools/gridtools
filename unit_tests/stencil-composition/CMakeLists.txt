# collect test cases
fetch_host_tests(.)
fetch_gpu_tests(.)

# collect the cache and redcution tests.
#those are only enabled when using structured grids
if (STRUCTURED_GRIDS)
    fetch_host_tests(caches)
    fetch_gpu_tests(caches)
    fetch_host_tests(reductions)
    fetch_gpu_tests(reductions)
endif()

# collect the grid specific test cases
if (NOT STRUCTURED_GRIDS)
    fetch_host_tests(icosahedral_grids)
    fetch_gpu_tests(icosahedral_grids)
    if( ENABLE_ATLAS )
        fetch_host_tests(icosahedral_grids/unstructured ${ATLAS_LIBRARIES})
        fetch_gpu_tests(icosahedral_grids/unstructured ${ATLAS_LIBRARIES})
    endif()
else()
    fetch_host_tests(structured_grids)
    fetch_gpu_tests(structured_grids)
    # multi test is a custom test case that tests multiple variations
    if( ENABLE_CXX11 )
        foreach( variation CALL OFFSETS PROCEDURES PROCEDURES_OFFSETS )
            string(TOLOWER ${variation} var_suffix)
            set(ccflags_block "-DFUNCTIONS_${variation} -DBACKEND_BLOCK")
            set(ccflags_naive "-DFUNCTIONS_${variation}")
            set(ldflags "${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS}")
            add_custom_host_test(multi_types_block_${var_suffix} structured_grids/multi_types_test.cpp ${ccflags_block} ${ldflags})
            add_custom_host_test(multi_types_naive_${var_suffix} structured_grids/multi_types_test.cpp ${ccflags_naive} ${ldflags})
        endforeach(variation)
    endif()
endif()
