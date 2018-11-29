# collect test cases
fetch_x86_tests(. LABELS unittest_x86 )
fetch_mc_tests(. LABELS unittest_mc)
fetch_gpu_tests(. LABELS unittest_cuda)

# collect structured grid specific tests
if(GT_TESTS_STRUCTURED_GRID)
    fetch_x86_tests(structured_grids LABELS unittest_x86)
    fetch_mc_tests(structured_grids LABELS unittest_mc)
    fetch_gpu_tests(structured_grids LABELS unittest_cuda)
endif()

# collect icosahedral grid specific tests
if (NOT GT_TESTS_STRUCTURED_GRID)
    fetch_x86_tests(icosahedral_grids LABELS unittest_x86)
    fetch_mc_tests(icosahedral_grids LABELS unittest_mc)
    fetch_gpu_tests(icosahedral_grids LABELS unittest_cuda)
endif()


