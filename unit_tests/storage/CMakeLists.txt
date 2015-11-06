
include_directories( ../../include/storage ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/storage/[^.]*.hpp")

SET(STORAGE_UNITTEST_SOURCES test_smallstorage_indices test_meta_storage test_data_fields test_meta_storage ${INC_ALL})

### Executables
#CPU tests
add_executable(storage_tests ${STORAGE_UNITTEST_SOURCES})
gridtools_add_test(tests.storage ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/storage_tests")

set_target_properties(storage_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )
target_link_libraries(storage_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

