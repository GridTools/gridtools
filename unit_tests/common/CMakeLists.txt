
include_directories( ../../include/common ) ## Library header files
### Executables
#CPU tests

add_executable(common_tests common_tests.cpp meta_array_test.cpp)
set_target_properties(common_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(common_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

add_test( common_tests ${CMAKE_BINARY_DIR}/unit_tests/build/${common_tests})
