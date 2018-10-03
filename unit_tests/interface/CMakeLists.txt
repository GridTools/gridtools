#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(repository)

# collect test cases
fetch_host_tests(.)
fetch_gpu_tests(.)
fetch_host_tests(./layout_transformation)
fetch_gpu_tests(./layout_transformation)
