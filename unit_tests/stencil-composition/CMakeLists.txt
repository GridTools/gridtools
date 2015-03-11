
include_directories( ../../include/stencil-composition ) ## Library header files
### Executables
#CPU tests
add_executable(stencil_composition_tests test_backend.cpp test_make_computation.cpp)
set_target_properties(stencil_composition_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(stencil_composition_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

add_test( common_tests ${CMAKE_BINARY_DIR}/unit_tests/build/${stencil-composition_tests})
