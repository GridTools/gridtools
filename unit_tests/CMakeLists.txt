include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(unittests_c_bindings_handle ${CMAKE_SOURCE_DIR}/src/c_bindings/handle.cpp)
add_library(unittests_fortran_bindings ${CMAKE_SOURCE_DIR}/src/c_bindings/array_descriptor.f90 ${CMAKE_SOURCE_DIR}/src/c_bindings/handle.f90)
target_link_libraries(unittests_fortran_bindings unittests_c_bindings_handle)

set(exe_LIBS ${exe_LIBS} unittests_fortran_bindings)
set(exe_LIBS ${exe_LIBS} c_bindings_generator)

# collect tests
fetch_x86_tests(.)
fetch_mc_tests(.)
fetch_gpu_tests(.)

# include all subdirs
add_subdirectory( common )
add_subdirectory( communication )
add_subdirectory( boundary-conditions )
add_subdirectory( interface )
add_subdirectory( distributed-boundaries )
add_subdirectory( stencil-composition )
add_subdirectory( storage )
add_subdirectory( c_bindings )

