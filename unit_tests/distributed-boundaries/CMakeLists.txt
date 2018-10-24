set(ccflags ${X86_SPECIFIC_OPTIONS} )
set(ldflags " ")
add_custom_x86_test(TARGET test_bindbc_utilities SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_bindbc_utilities.cpp ADDITIONAL_FLAGS ${ccflags} )

if( USE_MPI )

    add_custom_mpi_x86_test(distributed_boundaries ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cpp ${ccflags} ${ldflags})
    if ( USE_GPU )
        set(ccflags " ")
        set(ldflags " ")

        add_custom_mpi_gpu_test(distributed_boundaries_cuda ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cu ${ccflags} ${ldflags})
    endif()

else()

    add_custom_x86_test(TARGET distributed_boundaries SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cpp ADDITIONAL_FLAGS ${ccflags} )

    if ( USE_GPU )
        set(ccflags " ")
        set(ldflags " ")

        add_custom_gpu_test(distributed_boundaries_cuda ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_boundaries.cu ${ccflags} ${ldflags})
    endif()

endif()
