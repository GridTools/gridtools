FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/*.hpp")

SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${INC_ALL})
if ( ENABLE_CXX11 )
  SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${STENCIL_COMPOSITION_UNITTEST_SOURCES} test-assign-placeholders.cpp)

  ## moved here since there is only one test
  add_executable(stencil_composition_tests ${STENCIL_COMPOSITION_UNITTEST_SOURCES})
  gridtools_add_test(tests.stencil_composition ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/stencil_composition_tests")

  set_target_properties(stencil_composition_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(stencil_composition_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

endif( ENABLE_CXX11 )

### Executables
#CPU tests

add_subdirectory(caches)
if (NOT STRUCTURED_GRIDS)
  add_subdirectory(icosahedral_grids)
else()
  add_subdirectory(structured_grids)
endif()
