set(ccflags ${X86_SPECIFIC_OPTIONS} )

add_custom_x86_test(TARGET test_bindbc_utilities SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_bindbc_utilities.cpp ADDITIONAL_FLAGS ${ccflags} )

if( GT_USE_MPI )

    if( GT_ENABLE_X86 )
        add_custom_mpi_x86_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cpp )
        target_link_libraries(distributed_boundaries_x86 gcl)
    endif()

    if( GT_ENABLE_GPU )
        add_custom_mpi_gpu_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cu )
        target_link_libraries(distributed_boundaries_cuda gcl)
    endif()

else()

    if( GT_ENABLE_X86 )
        add_custom_x86_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cpp ADDITIONAL_FLAGS ${ccflags} )
        target_link_libraries(distributed_boundaries_x86 gcl)
    endif()

    if( GT_ENABLE_GPU )
        add_custom_gpu_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cu )
        target_link_libraries(distributed_boundaries_cuda gcl)
    endif()

endif()
