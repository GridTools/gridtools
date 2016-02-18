include_directories( ../../include/stencil-composition ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/[a-z]*.hpp")

set(RECGRID_TESTS ${INC_ALL})

if ( ENABLE_CXX11 )
    set(RECGRID_TESTS test_local_domain.cpp test_staggered_keyword.cpp test_make_computation.cpp test_cycle_and_swap.cpp test_iterate_domain.cpp test_mss_metafunctions.cpp)

    add_executable(strgrid_stencil_tests ${RECGRID_TESTS})
  # include boost
    include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
      link_directories( ${Boost_LIBRARY_DIRS} )

    set_target_properties(strgrid_stencil_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(strgrid_stencil_tests ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

    gridtools_add_test(strgrid_stencil_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests")

    add_executable(test_cycle_and_swap test_cycle_and_swap.cpp)
    set_target_properties(test_cycle_and_swap PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_cycle_and_swap ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

endif(ENABLE_CXX11)
if(CUDA_FOUND)
    if(ENABLE_CXX11)
        set(RECGRID_CUDA_TESTS test_cycle_and_swap.cu)

        cuda_add_executable(test_cycle_and_swap_cuda ${RECGRID_CUDA_TESTS})
        set_target_properties(test_cycle_and_swap_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
        target_link_libraries(test_cycle_and_swap_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

cuda_add_executable(strgrid_stencil_tests_cuda ${RECGRID_CUDA_TESTS})
        set_target_properties(strgrid_stencil_tests_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
        target_link_libraries(strgrid_stencil_tests_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
        gridtools_add_test(strgrid_stencil_tests_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests_cuda")
    endif()
endif()
