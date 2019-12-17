# collect tests
fetch_x86_tests(. LABELS unittest_x86)
fetch_gpu_tests(. LABELS unittest_cuda)

# include all subdirs
add_subdirectory(common)
add_subdirectory(communication)
add_subdirectory(boundary_conditions)
add_subdirectory(distributed_boundaries)
add_subdirectory(stencil_composition)
add_subdirectory(storage)
add_subdirectory(interface)

fetch_x86_tests(sid LABELS unittest_x86)
fetch_gpu_tests(sid LABELS unittest_cuda)

