add_subdirectory(repository)

# collect test cases
fetch_x86_tests(. LABELS unittest interface)
fetch_gpu_tests(. LABELS unittest interface)
fetch_x86_tests(./layout_transformation LABELS unittest interface)
fetch_gpu_tests(./layout_transformation LABELS unittest interface)
