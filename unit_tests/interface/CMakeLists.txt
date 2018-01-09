add_subdirectory(repository)

# collect test cases
fetch_host_tests(.)
fetch_gpu_tests(.)
