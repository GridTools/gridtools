include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# collect tests
fetch_host_tests(.)
fetch_gpu_tests(.)

# include all subdirs
add_subdirectory( common )
add_subdirectory( communication )
add_subdirectory( boundary-conditions )
add_subdirectory( distributed-boundaries )
add_subdirectory( stencil-composition )
add_subdirectory( storage )
