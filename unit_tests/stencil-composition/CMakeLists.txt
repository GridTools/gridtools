
include_directories( ../../include/stencil-composition ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/[a-z]*.hpp")

SET(STENCIL_COMPOSITION_UNITTEST_SOURCES test_make_computation.cpp test_accessor.cpp ${INC_ALL})
if ( ENABLE_CXX11 )
  SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${STENCIL_COMPOSITION_UNITTEST_SOURCES} test_cycle_and_swap.cpp test_iterate_domain.cpp test_mss_metafunctions.cpp)
endif( ENABLE_CXX11 )


### Executables
#CPU tests
add_executable(stencil_composition_tests ${STENCIL_COMPOSITION_UNITTEST_SOURCES})
gridtools_add_test(tests.stencil_composition ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/stencil_composition_tests")

set_target_properties(stencil_composition_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(stencil_composition_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

if ( ENABLE_CXX11 )
add_executable(test_cycle_and_swap test_cycle_and_swap.cpp)
set_target_properties(test_cycle_and_swap PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )
target_link_libraries(test_cycle_and_swap ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
gridtools_add_test(tests.test_cycle_and_swap ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_cycle_and_swap")

add_executable(test_iterate_domain test_iterate_domain.cpp)
set_target_properties(test_iterate_domain PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} LINKER_LANGUAGE CXX )
target_link_libraries(test_iterate_domain ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
gridtools_add_test(tests.test_iterate_domain ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_cycle_and_swap")

if (CUDA_FOUND)
cuda_add_executable(test_cycle_and_swap_cuda test_cycle_and_swap.cu)
target_link_libraries(test_cycle_and_swap_cuda ${GTEST_BOTH_LIBRARIES} gcl ${exe_LIBS})
gridtools_add_test(tests_gpu.test_cycle_and_swap ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_cycle_and_swap_cuda")
endif(CUDA_FOUND)

endif(ENABLE_CXX11)

add_subdirectory(caches)
