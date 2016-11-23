# collect all tests in current folder
fetch_host_tests(.)
fetch_gpu_tests(.)

# collect all tests in generic_metafunctions folder
fetch_host_tests(generic_metafunctions)
fetch_gpu_tests(generic_metafunctions)
