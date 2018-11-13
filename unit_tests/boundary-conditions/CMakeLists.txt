# collect test cases
fetch_x86_tests(.)
fetch_mc_tests(.)
fetch_gpu_tests(.)

# collect structured grid specific tests
if(STRUCTURED_GRIDS)
    fetch_x86_tests(structured_grids)
    fetch_mc_tests(structured_grids)
    fetch_gpu_tests(structured_grids)
endif()

# collect icosahedral grid specific tests
if (NOT STRUCTURED_GRIDS)
    fetch_x86_tests(icosahedral_grids)
    fetch_mc_tests(icosahedral_grids)
    fetch_gpu_tests(icosahedral_grids)
endif()


