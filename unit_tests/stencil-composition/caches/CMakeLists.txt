
include_directories( ../../../include/stencil-composition ) ## Library header files
include_directories( ../../../include/stencil-composition/caches ) ## Library header files
include_directories( ../../../include/common ) ## Library header files

### Executables
#CPU tests
if(STRUCTURED_GRIDS)
    set(TEST_SOURCES test_cache_stencil.cpp)
endif(STRUCTURED_GRIDS)

if ( ENABLE_CXX11 )
  set(TEST_SOURCES ${TEST_SOURCES} test_define_caches.cpp)
  if(STRUCTURED_GRIDS)
      set(TEST_SOURCES ${TEST_SOURCES} test_cache_metafunctions.cpp)
  endif(STRUCTURED_GRIDS)
endif( ENABLE_CXX11 )

add_executable(cache_tests ${TEST_SOURCES})
gridtools_add_test(tests.caches ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/cache_tests")

set_target_properties(cache_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(cache_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

if(USE_GPU AND STRUCTURED_GRIDS)
    set(CUDA_SOURCES test_cache_stencil.cu)
    cuda_add_executable(cache_tests_cuda ${CUDA_SOURCES})
    target_link_libraries(cache_tests_cuda ${exe_LIBS})

    gridtools_add_test(tests.caches_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/cache_tests_cuda")
endif(USE_GPU AND STRUCTURED_GRIDS)
