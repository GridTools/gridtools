# collect tests
fetch_host_tests(.)
fetch_gpu_tests(.)

# include all subdirs
add_subdirectory( common )
add_subdirectory( communication )
add_subdirectory( boundary-conditions )
add_subdirectory( stencil-composition )
