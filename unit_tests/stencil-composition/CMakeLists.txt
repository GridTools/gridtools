# collect test cases
fetch_x86_tests(. LABELS unittest_x86)
fetch_gpu_tests(. LABELS unittest_cuda)

fetch_x86_tests(sid LABELS unittest_x86)
fetch_gpu_tests(sid LABELS unittest_cuda)

# collect the grid specific test cases
if (GT_TESTS_STRUCTURED_GRID)
    fetch_mc_tests(. LABELS unittest_mc)

    # collect the cache redcution and conditionals tests.
    # those are only enabled when using structured grids
    fetch_x86_tests(caches LABELS unittest_x86)
    fetch_mc_tests(caches LABELS unittest_mc)
    fetch_gpu_tests(caches LABELS unittest_cuda)

    fetch_x86_tests(conditionals LABELS unittest_x86)
    fetch_mc_tests(conditionals LABELS unittest_mc)
    fetch_gpu_tests(conditionals LABELS unittest_cuda)

    fetch_x86_tests(structured_grids LABELS unittest_x86)
    fetch_mc_tests(structured_grids LABELS unittest_mc)
    fetch_gpu_tests(structured_grids LABELS unittest_cuda)

    # multi test is a custom test case that tests multiple variations
    foreach( variation CALL PROCEDURES )
        string(TOLOWER ${variation} var_suffix)
        add_custom_x86_test(
            TARGET multi_types_block_${var_suffix}
            SOURCES structured_grids/multi_types_test.cpp
            COMPILE_DEFINITIONS FUNCTIONS_${variation}
            )

        add_custom_x86_test(
            TARGET multi_types_naive_${var_suffix}
            SOURCES structured_grids/multi_types_test.cpp
            COMPILE_DEFINITIONS FUNCTIONS_${variation} GT_BACKEND_STRATEGY_NAIVE
            )

        add_custom_mc_test(
            TARGET multi_types_block_${var_suffix}
            SOURCES structured_grids/multi_types_test.cpp  
            COMPILE_DEFINITIONS FUNCTIONS_${variation}
            )
    endforeach(variation)

else()
    fetch_x86_tests(icosahedral_grids LABELS unittest_x86)
    fetch_gpu_tests(icosahedral_grids LABELS unittest_cuda)
endif()
