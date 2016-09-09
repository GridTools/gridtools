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
  
  add_executable(test_expressions test_expressions.cpp ${GTEST_FILES})
  gridtools_add_test(test_expressions ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_expressions")
  set_target_properties(test_expressions PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(test_expressions ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

endif( ENABLE_CXX11 )

if(CUDA_FOUND)
  if(ENABLE_CXX11)
    cuda_add_executable(test_expressions_cuda test_expressions.cu ${GTEST_FILES})
    gridtools_add_test(test_expressions_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_expressions_cuda")
    set_target_properties(test_expressions_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_expressions_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
  endif()
endif()


add_subdirectory(caches)
if (NOT STRUCTURED_GRIDS)
  add_subdirectory(icosahedral_grids)
else()
  add_subdirectory(structured_grids)
endif()
