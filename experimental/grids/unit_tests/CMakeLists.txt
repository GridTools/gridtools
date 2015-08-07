
include_directories( ../ ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../*.hpp")

SET(GRIDS_UNITTEST_SOURCES test_from_to_mapping.cpp ${INC_ALL})

### Executables
#CPU tests
add_executable(grids_tests ${GRIDS_UNITTEST_SOURCES})
add_test(tests.grids "${CMAKE_BINARY_DIR}/build/grids_tests")

set_target_properties(grids_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(grids_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

