# collect test cases
fetch_x86_tests(. LABELS unittest boundary_conditions)
fetch_mc_tests(. LABELS unittest boundary_conditions)
fetch_gpu_tests(. LABELS unittest boundary_conditions)

# collect structured grid specific tests
if(STRUCTURED_GRIDS)
    fetch_x86_tests(structured_grids LABELS unittest boundary_conditions)
    fetch_mc_tests(structured_grids LABELS unittest boundary_conditions)
    fetch_gpu_tests(structured_grids LABELS unittest boundary_conditions)
endif()

# collect icosahedral grid specific tests
if (NOT STRUCTURED_GRIDS)
    fetch_x86_tests(icosahedral_grids LABELS unittest boundary_conditions)
    fetch_mc_tests(icosahedral_grids LABELS unittest boundary_conditions)
    fetch_gpu_tests(icosahedral_grids LABELS unittest boundary_conditions)
endif()


