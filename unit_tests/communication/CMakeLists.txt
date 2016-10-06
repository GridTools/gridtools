# collect all tests in current folder
fetch_host_tests(.)
fetch_gpu_tests(.)

# custom test cases
if( NOT GCL_ONLY )
    if( GCL_MPI )
        set(ccflags " ${HOST_SPECIFIC_OPTIONS}")
        set(ldflags " ${MPI_ADDITIONAL_LIBS}")

set(COMMUNICATION_SOURCES
    halo_exchange_3D.cpp

    # unit test driver
    test_driver.cpp
)
add_executable(comm_tests ${COMMUNICATION_SOURCES})
target_link_libraries(comm_tests gcl gtest boost_system)
        ### # 3D MPI tests
        ### foreach(srcfile halo_exchange_3D ) # proc_grid_3D)
        ###     set(filename "${srcfile}.cpp")
        ###     add_custom_comm_host_test(${srcfile} ${filename} ${ccflags} ${ldflags})
        ### endforeach()

        ### # 2D MPI tests
        ### foreach(srcfile descriptors_3D empty_descriptors_3D boolist ndloops)
        ###     set(filename "${srcfile}.cpp")
        ###     add_custom_comm_host_test(${srcfile} ${filename} ${ccflags} ${ldflags})
        ### endforeach()
        endif()
endif()
