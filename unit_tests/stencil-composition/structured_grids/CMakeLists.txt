FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/[a-z|A-Z|0-9]*.hpp" "../../include/stencil-composition/structured_grids/[a-z|A-Z|0-9]*.hpp")

if( ENABLE_CXX11 )

        foreach( variation CALL OFFSETS PROCEDURES PROCEDURES_OFFSETS )
            string(TOLOWER ${variation} var_suffix)
            add_executable(multi_types_block_${var_suffix} multi_types_test.cpp)
            add_executable(multi_types_naive_${var_suffix} multi_types_test.cpp)
            set_target_properties(multi_types_block_${var_suffix} PROPERTIES COMPILE_FLAGS "${ADDITIONAL_CXX_FLAGS} -DFUNCTIONS_${variation} -DBACKEND_BLOCK")
            set_target_properties(multi_types_naive_${var_suffix} PROPERTIES COMPILE_FLAGS "${ADDITIONAL_CXX_FLAGS} -DFUNCTIONS_${variation}")
            target_link_libraries(multi_types_block_${var_suffix} ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
            target_link_libraries(multi_types_naive_${var_suffix} ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
            gridtools_add_test(tests.multi_types_block_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/multi_types_block_${var_suffix}")
            gridtools_add_test(tests.multi_types_naive_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/multi_types_naive_${var_suffix}")
        endforeach(variation)
endif()

if(CUDA_FOUND)
     if( ENABLE_CXX11 )
        foreach( variation CALL OFFSETS PROCEDURES PROCEDURES_OFFSETS)
            string(TOLOWER ${variation} var_suffix)
            cuda_add_executable(multi_types_cuda_${var_suffix} multi_types_test.cu OPTIONS "-DFUNCTIONS_${variation}")
            target_link_libraries(multi_types_cuda_${var_suffix} ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
            gridtools_add_test(tests.multi_types_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/multi_types_cuda_${var_suffix}")
        endforeach(variation)
     endif()
endif()


    add_executable(test_conditionals test_conditionals.cpp ${INC_ALL})
    set_target_properties(test_conditionals PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_conditionals ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
    gridtools_add_test(test_conditionals ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_conditionals")

if ( ENABLE_CXX11 )
    add_executable(test_multiple_output_stencils test_multiple_output_stencils.cpp)
    set_target_properties(test_multiple_output_stencils PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_multiple_output_stencils ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
    gridtools_add_test(test_multiple_output_stencils ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_conditionals")


    add_executable(test_conditional_switches test_conditional_switches.cpp)
    set_target_properties(test_conditional_switches PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_conditional_switches ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
    gridtools_add_test(test_conditional_switches ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_conditional_switches")

    set(RECGRID_TESTS test_local_domain.cpp test_staggered_keyword.cpp test_make_computation.cpp test_cycle_and_swap.cpp test_iterate_domain.cpp test_mss_metafunctions.cpp call_interfaces_metafunctions.cpp test_esf_metafunctions.cpp  test_accessor.cpp positional_when_debug.cpp test-unfold-independent.cpp test-unfold-all.cpp test_read_write_esf.cpp )

    add_executable(strgrid_stencil_tests ${RECGRID_TESTS})
    # include boost
    #WHYYYYYYYYY>??
    include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
    link_directories( ${Boost_LIBRARY_DIRS} )

    set_target_properties(strgrid_stencil_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(strgrid_stencil_tests ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

    gridtools_add_test(strgrid_stencil_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests")


  add_executable(test_domain_reassign test_domain_reassign.cpp ${GTEST_FILES})
  gridtools_add_test(test_domain_reassign ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_domain_reassign")

  set_target_properties(test_domain_reassign PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(test_domain_reassign ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

endif(ENABLE_CXX11)
if(CUDA_FOUND)
    if(ENABLE_CXX11)
        set(RECGRID_CUDA_TESTS test_cycle_and_swap.cu test_make_computation.cu positional_when_debug.cu)

        cuda_add_executable(strgrid_stencil_tests_cuda ${RECGRID_CUDA_TESTS})
        set_target_properties(strgrid_stencil_tests_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
        target_link_libraries(strgrid_stencil_tests_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
        gridtools_add_test(strgrid_stencil_tests_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests_cuda")
    endif()

  cuda_add_executable(test_conditionals_cuda test_conditionals.cu ${GTEST_FILES})
  gridtools_add_test(test_conditionals_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_conditionals_cuda")
  set_target_properties(test_conditionals_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(test_conditionals_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

if(CUDA_VERSION_MAJOR>7)
    cuda_add_executable(test_conditional_switches_cuda test_conditional_switches.cu)
    set_target_properties(test_conditional_switches_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_conditional_switches_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
    gridtools_add_test(test_conditional_switches_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_conditional_switches_cuda")
endif(CUDA_VERSION_MAJOR>7)

if (CUDA_FOUND)
  if(ENABLE_CXX11)
        cuda_add_executable(test_domain_reassign_cuda test_domain_reassign.cu ${GTEST_FILES})
  gridtools_add_test(test_domain_reassign_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_domain_reassign_cuda")
    set_target_properties(test_domain_reassign_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_domain_reassign_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
  endif(ENABLE_CXX11)
endif(CUDA_FOUND)

endif()
