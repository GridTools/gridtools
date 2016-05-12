
include_directories( ../../include/common ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/common/*.hpp")

SET(COMMON_UNITTEST_SOURCES meta_array_test.cpp ${INC_ALL})


### Executables
#CPU tests

add_executable(common_tests ${COMMON_UNITTEST_SOURCES})

set_target_properties(common_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(common_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

gridtools_add_test( tests.common_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/common_tests")

add_subdirectory( generic_metafunctions )