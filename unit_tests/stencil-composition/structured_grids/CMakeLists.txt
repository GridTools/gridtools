FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/*.hpp" "../../include/stencil-composition/structured_grids/*.hpp")

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


if ( ENABLE_CXX11 )
    set(RECGRID_TESTS test_local_domain.cpp test_staggered_keyword.cpp test_make_computation.cpp test_cycle_and_swap.cpp test_iterate_domain.cpp test_mss_metafunctions.cpp call_interfaces_metafunctions.cpp test_esf_metafunctions.cpp  test_accessor.cpp positional_when_debug.cpp)

    add_executable(strgrid_stencil_tests ${RECGRID_TESTS})
    # include boost
    #WHYYYYYYYYY>??
    include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
    link_directories( ${Boost_LIBRARY_DIRS} )

    set_target_properties(strgrid_stencil_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(strgrid_stencil_tests ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

    gridtools_add_test(strgrid_stencil_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests")

    add_executable(test_cycle_and_swap test_cycle_and_swap.cpp)
    set_target_properties(test_cycle_and_swap PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_cycle_and_swap ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

    add_executable(test_iterate_domain test_iterate_domain.cpp)
    set_target_properties(test_iterate_domain PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_iterate_domain ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

    add_executable(test_conditional_switches test_conditional_switches.cpp)
    set_target_properties(test_conditional_switches PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_conditional_switches ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

    ## # add_executable(test_conditionals test_conditionals.cpp)
    ## # set_target_properties(test_conditionals PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    ## # target_link_libraries(test_conditionals ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

endif(ENABLE_CXX11)
if(CUDA_FOUND)
    if(ENABLE_CXX11)
        #set(RECGRID_CUDA_TESTS test_cycle_and_swap.cu test_make_computation.cu positional_when_debug.cu)
        set(RECGRID_CUDA_TESTS positional_when_debug.cu)

        cuda_add_executable(strgrid_stencil_tests_cuda ${RECGRID_CUDA_TESTS})
        set_target_properties(strgrid_stencil_tests_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
        target_link_libraries(strgrid_stencil_tests_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
        gridtools_add_test(strgrid_stencil_tests_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests_cuda")
    endif()
endif()
