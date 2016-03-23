
include_directories( ../../include/reductions ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/reductions/[a-z]*.hpp")

SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${INC_ALL})
if (ENABLE_CXX11  AND STRUCTURED_GRIDS)
  SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${STENCIL_COMPOSITION_UNITTEST_SOURCES} test_make_reductions.cpp)

  ### Executables
  #CPU tests
  add_executable(reductions_tests ${STENCIL_COMPOSITION_UNITTEST_SOURCES})
  gridtools_add_test(tests.reductions ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/reductions_tests")

  set_target_properties(reductions_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(reductions_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

endif(ENABLE_CXX11 AND STRUCTURED_GRIDS)



