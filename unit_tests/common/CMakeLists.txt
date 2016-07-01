include_directories( ../../include/common ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/common/*.hpp *.hpp")

SET(COMMON_UNITTEST_SOURCES meta_array_test.cpp atomic_functions_test.cpp test_layout_map.cpp test_offset_tuple.cpp ${INC_ALL})

if(ENABLE_CXX11)
    SET(COMMON_UNITTEST_SOURCES ${COMMON_UNITTEST_SOURCES} test_array.cpp test_aggregate.cpp explode_array_test.cpp test_repeat_template.cpp test_tuple.cpp)
endif()

### Executables
#CPU tests

add_executable(common_tests ${COMMON_UNITTEST_SOURCES})

set_target_properties(common_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(common_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

gridtools_add_test( tests.common_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/common_tests")

if (CUDA_FOUND)
    set(COMMON_UNITTEST_SOURCES_CUDA atomic_functions_test.cu test_offset_tuple.cu test_layout_map.cu test_tuple.cu)
    if(ENABLE_CXX11)
        set(COMMON_UNITTEST_SOURCES_CUDA ${COMMON_UNITTEST_SOURCES_CUDA} explode_array_test.cu)
    endif()
    cuda_add_executable(common_tests_cuda ${COMMON_UNITTEST_SOURCES_CUDA})
    gridtools_add_test(tests.common_tests_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/common_tests_cuda")
    target_link_libraries(common_tests_cuda ${GTEST_BOTH_LIBRARIES} ${exe_LIBS})

endif()


add_subdirectory( generic_metafunctions )
