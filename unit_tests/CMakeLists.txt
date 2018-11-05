
add_library(unittests_c_bindings_handle ${CMAKE_SOURCE_DIR}/src/c_bindings/handle.cpp)
target_include_directories(unittests_c_bindings_handle PUBLIC ${CMAKE_SOURCE_DIR}/include)
add_library(unittests_fortran_bindings ${CMAKE_SOURCE_DIR}/src/c_bindings/array_descriptor.f90 ${CMAKE_SOURCE_DIR}/src/c_bindings/handle.f90)
target_link_libraries(unittests_fortran_bindings unittests_c_bindings_handle)

set(exe_LIBS ${exe_LIBS} unittests_fortran_bindings)
set(exe_LIBS ${exe_LIBS} c_bindings_generator)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# collect tests
fetch_x86_tests(.)
fetch_mc_tests(.)
fetch_gpu_tests(.)

# include all subdirs
if ( COMPONENT_COMMON )
   add_subdirectory( common )
endif()
if ( COMPONENT_GCL )
   add_subdirectory( communication )
endif()
if ( COMPONENT_BOUNDARY_CONDITIONS )
   add_subdirectory( boundary-conditions )
endif()

if ( COMPONENT_DISTRIBUTED_BOUNDARIES )
   add_subdirectory( distributed-boundaries )
endif()
if ( COMPONENT_STENCIL_COMPOSITION )
   add_subdirectory( stencil-composition )
endif()
if ( COMPONENT_STORAGE )
   add_subdirectory( storage )
endif()

if ( COMPONENT_C_BINDINGS )
   add_subdirectory( interface )
   add_subdirectory( c_bindings )
endif()
