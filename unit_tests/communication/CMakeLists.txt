# collect all tests in current folder
fetch_host_tests(.)
fetch_gpu_tests(.)

# custom test cases 
if( NOT GCL_ONLY )
    if( GCL_MPI )
        set(ccflags " ${HOST_SPECIFIC_OPTIONS}")
        set(ldflags " ${MPI_ADDITIONAL_LIBS}")

        # 3D MPI tests
#        foreach(srcfile halo_exchange_3D proc_grid_3D)
#            set(filename "${srcfile}.cpp")
#            add_custom_host_test(${srcfile} ${filename} ${ccflags} ${ldflags})
#        endforeach()

        # 2D MPI tests
# FIXME (deactivated due to issue #439): descriptors_3D         
        foreach(srcfile empty_descriptors_3D boolist ndloops)
            set(filename "${srcfile}.cpp")
            add_custom_host_test(${srcfile} ${filename} ${ccflags} ${ldflags})
        endforeach()
        endif()
endif()
