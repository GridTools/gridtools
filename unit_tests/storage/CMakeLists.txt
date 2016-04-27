
include_directories( ../../include/storage ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/storage/[^.]*.hpp")

SET(STORAGE_UNITTEST_SOURCES test_meta_storage test_data_fields test_meta_storage test_storage_naming ${INC_ALL})

### Executables
#CPU tests
add_executable(storage_tests ${STORAGE_UNITTEST_SOURCES})
gridtools_add_test(tests.storage ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/storage_tests")
add_executable(storage_info_test test_meta_storage.cpp)
gridtools_add_test(tests.storage_info ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/storage_info_test")
add_executable(storage_alignment_test alignment.cpp)
gridtools_add_test(tests.storage_alignment ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/storage_alignment_test")
add_executable(storage_naming_test test_storage_naming.cpp)
gridtools_add_test(tests.storage_naming ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/storage_naming_test")

set_target_properties(storage_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )
set_target_properties(storage_info_test PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )
set_target_properties(storage_alignment_test PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )
set_target_properties(storage_naming_test PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )

target_link_libraries(storage_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
target_link_libraries(storage_info_test ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
target_link_libraries(storage_alignment_test ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
target_link_libraries(storage_naming_test ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

if( USE_GPU )
  gridtools_add_test(tests_gpu.storage_alignment ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/storage_alignment_test_cuda")
  cuda_add_executable(storage_alignment_test_cuda alignment.cu)
  set_target_properties(storage_alignment_test_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )
  target_link_libraries(storage_alignment_test_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

  gridtools_add_test(tests_gpu.storage_info_ptr_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/storage_info_ptr_cuda")
  cuda_add_executable(storage_info_ptr_cuda test_storage_info_gpu_pointer.cu)
  set_target_properties(storage_info_ptr_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )
  target_link_libraries(storage_info_ptr_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
endif (USE_GPU)
