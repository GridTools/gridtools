# collect all tests in current folder
fetch_host_tests(.)
fetch_gpu_tests(.)


add_custom_host_test(boolist boolist.cpp ${CMAKE_CXX_FLAGS} ${exe_LIBS})
add_custom_host_test(ndloops ndloops.cpp ${CMAKE_CXX_FLAGS} ${exe_LIBS})

# custom test cases
if( NOT GCL_ONLY )
    if( GCL_MPI )
        set(ccflags " ${HOST_SPECIFIC_OPTIONS}")
        set(ldflags " ${MPI_ADDITIONAL_LIBS}")
message ("CAZZO" ${ccflags})
        add_custom_mpi_host_test(halo_exchange_3D halo_exchange_3D.cpp ${ccflags} ${ldflags})
        add_custom_mpi_host_test(halo_exchange_3D_all ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cpp ${ccflags} ${ldflags})
        add_custom_mpi_host_test(halo_exchange_3D_all_2 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cpp ${ccflags} ${ldflags})
        add_custom_mpi_host_test(halo_exchange_3D_all_3 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cpp "-DUSELESS" ${exe_LIBS})
        add_custom_mpi_host_test(halo_exchange_3D_generic ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cpp ${ccflags} ${ldflags})
        add_custom_mpi_host_test(halo_exchange_3D_generic_full ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cpp ${ccflags} ${ldflags})

        if ( USE_GPU )
            set(ccflags " ${HOST_SPECIFIC_OPTIONS}")
            set(ldflags " ${MPI_ADDITIONAL_LIBS}")

            add_custom_mpi_gpu_test(cuda_halo_exchange_3D_all ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(cuda_halo_exchange_3D_all_2 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(cuda_halo_exchange_3D_all_3 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(cuda_halo_exchange_3D_generic ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(cuda_halo_exchange_3D_generic_full ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cu ${ccflags} ${ldflags})

              ### set(CUDA_COMMUNICATION_SOURCES
              ###      #halo_exchange_3D.cpp
              ###      ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cu
              ###      ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cu
              ###      ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cu
              ###      ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cu
              ###      ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cu

              ###      # unit test driver
              ###      test_driver.cpp
              ### )
              ### cuda_add_executable(cuda_comm_tests ${CUDA_COMMUNICATION_SOURCES} OPTIONS -I${MPI_INCLUDE_PATH})
              ### set_target_properties(cuda_comm_tests PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS}  )
              ### target_link_libraries(cuda_comm_tests ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS} gtest boost_system)
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
