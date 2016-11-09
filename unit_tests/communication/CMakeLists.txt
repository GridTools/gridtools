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
            ../../examples/communication/test_halo_exchange_3D_all.cpp
            ../../examples/communication/test_halo_exchange_3D_all_2.cpp
            ../../examples/communication/test_halo_exchange_3D_all_3.cpp

            # unit test driver
            test_driver.cpp
        )
        add_executable(comm_tests ${COMMUNICATION_SOURCES})
        target_link_libraries(comm_tests gcl gtest boost_system)

        if ( USE_GPU )
            set(ccflags " ${HOST_SPECIFIC_OPTIONS}")
              set(ldflags " ${MPI_ADDITIONAL_LIBS}")

              set(CUDA_COMMUNICATION_SOURCES
                   #halo_exchange_3D.cpp
                   ../../examples/communication/test_halo_exchange_3D_all.cu
                   ../../examples/communication/test_halo_exchange_3D_all_2.cu
                   ../../examples/communication/test_halo_exchange_3D_all_3.cu

                   # unit test driver
                   test_driver.cpp
              )
              cuda_add_executable(cuda_comm_tests ${CUDA_COMMUNICATION_SOURCES} OPTIONS -I${MPI_INCLUDE_PATH})
              set_target_properties(cuda_comm_tests PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS}  )
              target_link_libraries(cuda_comm_tests ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS} gtest boost_system)
        endif()

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
