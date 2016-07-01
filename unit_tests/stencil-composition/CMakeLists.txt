FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/[a-z|A-Z|0-9]*.hpp")

SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${INC_ALL})
if ( ENABLE_CXX11 )
  SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${STENCIL_COMPOSITION_UNITTEST_SOURCES} test-assign-placeholders.cpp )

  set( GTEST_FILES ${CMAKE_SOURCE_DIR}/examples/Options.hpp ${CMAKE_SOURCE_DIR}/examples/Options.cpp)

  ## moved here since there is only one test
  add_executable(stencil_composition_tests ${STENCIL_COMPOSITION_UNITTEST_SOURCES})
  gridtools_add_test(tests.stencil_composition ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/stencil_composition_tests")

  set_target_properties(stencil_composition_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(stencil_composition_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

  add_executable(test_domain_reassign test_domain_reassign.cpp ${GTEST_FILES})
  gridtools_add_test(test_domain_reassign ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_domain_reassign")

  set_target_properties(test_domain_reassign PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(test_domain_reassign ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

if (CUDA_FOUND)
  cuda_add_executable(test_domain_reassign_cuda test_domain_reassign.cu ${GTEST_FILES})
  gridtools_add_test(test_domain_reassign_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_domain_reassign_cuda")
  set_target_properties(test_domain_reassign_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(test_domain_reassign_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
endif(CUDA_FOUND)

endif( ENABLE_CXX11 )

### Executables
#CPU tests

add_subdirectory(caches)
if (NOT STRUCTURED_GRIDS)
  add_subdirectory(icosahedral_grids)
else()
  add_subdirectory(structured_grids)
endif()
