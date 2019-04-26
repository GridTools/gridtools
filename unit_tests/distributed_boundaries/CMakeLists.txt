add_custom_test(x86 TARGET test_bindbc_utilities SOURCES test_bindbc_utilities.cpp  )

if( GT_USE_MPI )

    if( GT_ENABLE_BACKEND_X86 )
        add_custom_mpi_test(
            x86
            TARGET test_distributed_boundaries
            NPROC 4
            SOURCES test_distributed_boundaries.cpp
            LABELS mpitest_x86)
        target_link_libraries(test_distributed_boundaries_x86 gcl)
    endif()

    if( GT_ENABLE_BACKEND_CUDA )
        add_custom_mpi_test(
            cuda
            TARGET test_distributed_boundaries
            NPROC 4
            SOURCES test_distributed_boundaries.cu
            LABELS mpitest_cuda)
        target_link_libraries(test_distributed_boundaries_cuda gcl)
    endif()

else()

    if( GT_ENABLE_BACKEND_X86 )
        add_custom_test(
            x86
            TARGET test_distributed_boundaries
            SOURCES test_distributed_boundaries.cpp
            LABELS mpitest_x86
            )
        target_link_libraries(test_distributed_boundaries_x86 gcl)
    endif()

    if( GT_ENABLE_BACKEND_CUDA )
        add_custom_test(
            cuda
            TARGET test_distributed_boundaries
            SOURCES test_distributed_boundaries.cu
            LABELS mpitest_cuda
            )
        target_link_libraries(test_distributed_boundaries_cuda gcl)
    endif()

endif()
