include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# collect all tests in current folder
fetch_x86_tests(.)
fetch_mc_tests(.)
fetch_gpu_tests(.)

# collect all tests in generic_metafunctions folder
fetch_x86_tests(generic_metafunctions)
fetch_mc_tests(generic_metafunctions)
fetch_gpu_tests(generic_metafunctions)
