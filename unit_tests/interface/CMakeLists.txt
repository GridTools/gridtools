add_subdirectory(repository)

# collect test cases
fetch_x86_tests(.)
fetch_gpu_tests(.)
fetch_x86_tests(./layout_transformation)
fetch_gpu_tests(./layout_transformation)
