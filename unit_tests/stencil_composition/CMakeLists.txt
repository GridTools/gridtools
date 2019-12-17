# collect test cases
fetch_x86_tests(. LABELS unittest_x86)
fetch_naive_tests(. LABELS unittest_naive)
fetch_gpu_tests(. LABELS unittest_cuda)
if(NOT GT_TESTS_ICOSAHEDRAL_GRID)
    fetch_mc_tests(. LABELS unittest_mc)
endif()

# collect the grid specific test cases
if (NOT GT_TESTS_ICOSAHEDRAL_GRID)
    # collect the cache redcution and conditionals tests.
    # those are only enabled when using structured grids
    fetch_x86_tests(caches LABELS unittest_x86)
    fetch_naive_tests(caches LABELS unittest_naive)
    fetch_mc_tests(caches LABELS unittest_mc)
    fetch_gpu_tests(caches LABELS unittest_cuda)

    fetch_x86_tests(conditionals LABELS unittest_x86)
    fetch_naive_tests(conditionals LABELS unittest_naive)
    fetch_mc_tests(conditionals LABELS unittest_mc)
    fetch_gpu_tests(conditionals LABELS unittest_cuda)

    fetch_x86_tests(structured_grids LABELS unittest_x86)
    fetch_naive_tests(structured_grids LABELS unittest_naive)
    fetch_mc_tests(structured_grids LABELS unittest_mc)
    fetch_gpu_tests(structured_grids LABELS unittest_cuda)

else()
    fetch_x86_tests(icosahedral_grids LABELS unittest_x86)
    fetch_naive_tests(icosahedral_grids LABELS unittest_naive)
    fetch_gpu_tests(icosahedral_grids LABELS unittest_cuda)
endif()
