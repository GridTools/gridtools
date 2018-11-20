add_subdirectory(repository)

# collect test cases
fetch_x86_tests(. LABELS unittest_x86 )
fetch_gpu_tests(. LABELS unittest_cuda )
fetch_x86_tests(./layout_transformation LABELS unittest_x86)
fetch_gpu_tests(./layout_transformation LABELS unittest_cuda)
