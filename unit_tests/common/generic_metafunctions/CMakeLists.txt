include_directories( ../../include/stencil-composition ) ## Library header files

file(GLOB_RECURSE INC_ALL "../../include/common/generic_metafunctions/[a-z]*.hpp")

set(GENERICMETA_UNITTEST_SOURCES test_is_there_in_sequence_if.cpp test_is_not_same.cpp test_apply_to_sequence.cpp ${INC_ALL})
if ( ENABLE_CXX11 )
    set(GENERICMETA_UNITTEST_SOURCES ${GENERICMETA_UNITTEST_SOURCES} test_integer_sequence.cpp variadic_typedef_test.cpp test_accumulate.cpp unzip.cpp)
endif( ENABLE_CXX11 )

add_executable(generic_metafunctions ${GENERICMETA_UNITTEST_SOURCES})

set_target_properties(generic_metafunctions PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(generic_metafunctions ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

gridtools_add_test( tests.generic_metafunctions ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/generic_metafunctions")

if (CUDA_FOUND)
    if( ENABLE_CXX11 )
        cuda_add_executable(generic_metafunctions_cuda test_accumulate.cu)
        gridtools_add_test(tests.test_accumulate ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/common_tests_cuda")
        target_link_libraries(generic_metafunctions_cuda ${GTEST_BOTH_LIBRARIES} ${exe_LIBS})
        gridtools_add_test( generic_metafunctions_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/generic_metafunctions_cuda")
    endif( ENABLE_CXX11 )
endif()
