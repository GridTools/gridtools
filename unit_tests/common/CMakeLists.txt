
include_directories( ../../include/common ) ## Library header files
### Executables
#CPU tests

add_executable(common_tests meta_array_test.cpp)

set_target_properties(common_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(common_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

gridtools_add_test( tests.common_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/common_tests")
