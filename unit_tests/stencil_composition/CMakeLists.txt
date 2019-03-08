# collect test cases
fetch_x86_tests(. LABELS unittest_x86)
fetch_gpu_tests(. LABELS unittest_cuda)

fetch_x86_tests(sid LABELS unittest_x86)
fetch_gpu_tests(sid LABELS unittest_cuda)

# collect the grid specific test cases
if (NOT GT_TESTS_ICOSAHEDRAL_GRID)
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

else()
    fetch_x86_tests(icosahedral_grids LABELS unittest_x86)
    fetch_gpu_tests(icosahedral_grids LABELS unittest_cuda)
endif()
