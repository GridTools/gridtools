
include_directories( ../../include/storage ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/storage/*.hpp")

SET(STENCIL_COMPOSITION_UNITTEST_SOURCES test_smallstorage_indices ${INC_ALL})

### Executables
#CPU tests
add_executable(storage_tests ${STENCIL_COMPOSITION_UNITTEST_SOURCES})
gridtools_add_test(tests.storage ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/storage_tests")

set_target_properties(storage_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(storage_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
