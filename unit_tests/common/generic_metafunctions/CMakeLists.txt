
### Executables
#CPU tests

add_executable(generic_metafunctions test_is_there_in_sequence_if.cpp test_is_not_same.cpp)

set_target_properties(generic_metafunctions PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(generic_metafunctions ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

gridtools_add_test( tests.generic_metafunctions ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/generic_metafunctions")
