set(ccflags ${X86_SPECIFIC_OPTIONS} )

add_custom_x86_test(TARGET test_bindbc_utilities SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_bindbc_utilities.cpp ADDITIONAL_FLAGS ${ccflags} )

if( USE_MPI )

    add_custom_mpi_x86_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cpp )
    target_link_libraries(distributed_boundaries_x86 gcl)
    if ( USE_GPU )
        add_custom_mpi_gpu_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cu )
        target_link_libraries(distributed_boundaries_cuda gcl)
    endif()

else()

    add_custom_x86_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cpp ADDITIONAL_FLAGS ${ccflags} )
    target_link_libraries(distributed_boundaries_x86 gcl)

    if ( USE_GPU )
        add_custom_gpu_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cu )
        target_link_libraries(distributed_boundaries_cuda gcl)
    endif()

endif()
