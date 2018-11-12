# collect test cases
fetch_x86_tests(. LABELS unittest stencil-composition)
fetch_mc_tests(. LABELS unittest stencil-composition)
fetch_gpu_tests(. LABELS unittest stencil-composition)

# collect the cache redcution and conditionals tests.
#those are only enabled when using structured grids
if (STRUCTURED_GRIDS)
    fetch_x86_tests(caches LABELS unittest stencil-composition)
    fetch_mc_tests(caches LABELS unittest stencil-composition)
    fetch_gpu_tests(caches LABELS unittest stencil-composition)

    fetch_x86_tests(reductions LABELS unittest stencil-composition)
    fetch_mc_tests(reductions LABELS unittest stencil-composition)
    fetch_gpu_tests(reductions LABELS unittest stencil-composition)

    fetch_x86_tests(conditionals LABELS unittest stencil-composition)
    fetch_mc_tests(conditionals LABELS unittest stencil-composition)
    fetch_gpu_tests(conditionals LABELS unittest stencil-composition)
endif()

# collect the grid specific test cases
if (NOT STRUCTURED_GRIDS)
    fetch_x86_tests(icosahedral_grids LABELS unittest stencil-composition)
    fetch_gpu_tests(icosahedral_grids LABELS unittest stencil-composition)

else()
    fetch_x86_tests(structured_grids LABELS unittest stencil-composition)
    fetch_mc_tests(structured_grids LABELS unittest stencil-composition)
    fetch_gpu_tests(structured_grids LABELS unittest stencil-composition)

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
            COMPILE_DEFINITIONS FUNCTIONS_${variation} BACKEND_STRATEGY_NAIVE
            )

        add_custom_mc_test(
            TARGET multi_types_block_${var_suffix}
            SOURCES structured_grids/multi_types_test.cpp  
            COMPILE_DEFINITIONS FUNCTIONS_${variation}
            )
    endforeach(variation)
endif()
