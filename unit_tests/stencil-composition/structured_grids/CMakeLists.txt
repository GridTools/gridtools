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


    add_executable(test_conditionals test_conditionals.cpp)
    set_target_properties(test_conditionals PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_conditionals ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
    gridtools_add_test(test_conditionals ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_conditionals")

if ( ENABLE_CXX11 )

    add_executable(test_conditional_switches test_conditional_switches.cpp)
    set_target_properties(test_conditional_switches PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test_conditional_switches ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
    gridtools_add_test(test_conditional_switches ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test_conditional_switches")

# TODO: removed from test since it breaks gtest by defining its own main(): test_esf_metafunctions.cpp 
    set(RECGRID_TESTS test_local_domain.cpp test_staggered_keyword.cpp test_make_computation.cpp test_cycle_and_swap.cpp test_iterate_domain.cpp test_mss_metafunctions.cpp call_interfaces_metafunctions.cpp test_accessor.cpp positional_when_debug.cpp test-unfold-independent.cpp test-unfold-all.cpp test_read_write_esf.cpp test_call_interfaces.cpp)

    add_executable(strgrid_stencil_tests ${RECGRID_TESTS})
    # include boost
    #WHYYYYYYYYY>??
    include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
    link_directories( ${Boost_LIBRARY_DIRS} )

    set_target_properties(strgrid_stencil_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(strgrid_stencil_tests ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

    gridtools_add_test(strgrid_stencil_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests")

endif(ENABLE_CXX11)
if(CUDA_FOUND)
    if(ENABLE_CXX11)
        set(RECGRID_CUDA_TESTS test_cycle_and_swap.cu test_make_computation.cu positional_when_debug.cu test_call_interfaces.cu )

        cuda_add_executable(strgrid_stencil_tests_cuda ${RECGRID_CUDA_TESTS})
        set_target_properties(strgrid_stencil_tests_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
        target_link_libraries(strgrid_stencil_tests_cuda ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
        gridtools_add_test(strgrid_stencil_tests_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests_cuda")
    endif()
endif()
