
include_directories( ../../../include/stencil-composition ) ## Library header files
include_directories( ../../../include/stencil-composition/caches ) ## Library header files
include_directories( ../../../include/common ) ## Library header files

### Executables
#CPU tests
if ( ENABLE_CXX11 )
  add_executable(cache_tests test_define_caches.cpp)
  gridtools_add_test(tests.caches ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/cache_tests")

  set_target_properties(cache_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(cache_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
endif( ENABLE_CXX11 )

