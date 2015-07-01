
include_directories( ../../include/stencil-composition ) ## Library header files
### Executables
#CPU tests
add_executable(stencil_composition_tests test_make_computation.cpp test_accessor.cpp)
set_target_properties(stencil_composition_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(stencil_composition_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

if ( ENABLE_CXX11 )
add_executable(test_iterate_domain test_iterate_domain.cpp)
set_target_properties(test_iterate_domain PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(test_iterate_domain ${exe_LIBS})
endif( ENABLE_CXX11 )

add_test( common_tests ${CMAKE_BINARY_DIR}/unit_tests/build/${stencil-composition_tests})
