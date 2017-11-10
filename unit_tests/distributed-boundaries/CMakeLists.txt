if( USE_MPI )
    set(ccflags " ${HOST_SPECIFIC_OPTIONS}" )
    set(ldflags " ")
    add_custom_mpi_host_test(distributed_boundaries ${CMAKE_CURRENT_SOURCE_DIR}/test_cxx11_distributed_boundaries.cpp ${ccflags} ${ldflags})
    if ( USE_GPU )
        set(ccflags " ")
        set(ldflags " ")

        add_custom_mpi_gpu_test(distributed_boundaries_cuda ${CMAKE_CURRENT_SOURCE_DIR}/test_cxx11_distributed_boundaries.cu ${ccflags} ${ldflags})
    endif()

endif()
