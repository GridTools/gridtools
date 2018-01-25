# collect all tests in current folder
fetch_host_tests(.)
fetch_mic_tests(.)
fetch_gpu_tests(.)


add_custom_host_test(boolist boolist.cpp ${CMAKE_CXX_FLAGS} ${exe_LIBS})
add_custom_host_test(ndloops ndloops.cpp ${CMAKE_CXX_FLAGS} ${exe_LIBS})
add_custom_mic_test(boolist boolist.cpp ${CMAKE_CXX_FLAGS} ${exe_LIBS})
add_custom_mic_test(ndloops ndloops.cpp ${CMAKE_CXX_FLAGS} ${exe_LIBS})

# custom test cases
if( NOT GCL_ONLY )
    if( USE_MPI )
        set(ccflags " ${HOST_SPECIFIC_OPTIONS}" )
        set(ldflags " ")
        
        if( NOT "${DISABLE_MPI_TESTS_ON_TARGET}" STREQUAL "CPU") 
            add_custom_mpi_host_test(halo_exchange_3D halo_exchange_3D.cpp ${ccflags} ${ldflags})

            add_custom_mpi_host_test(all_to_all_halo_3D ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_all_to_all_halo_3D.cpp ${ccflags} ${ldflags})

            add_custom_mpi_host_test(halo_exchange_3D_all ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cpp ${ccflags} ${ldflags})
            add_custom_mpi_host_test(halo_exchange_3D_all_2 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cpp ${ccflags} ${ldflags})
            add_custom_mpi_host_test(halo_exchange_3D_all_3 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cpp ${ccflags} ${ldflags})
            add_custom_mpi_host_test(halo_exchange_3D_generic ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cpp ${ccflags} ${ldflags})
            add_custom_mpi_host_test(halo_exchange_3D_generic_full ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cpp ${ccflags} ${ldflags})

            set(ccflags " ${HOST_SPECIFIC_OPTIONS} -DVECTOR_INTERFACE" )
            set(ldflags " ")

            add_custom_mpi_host_test(halo_exchange_3D_all_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cpp ${ccflags} ${ldflags})
            add_custom_mpi_host_test(halo_exchange_3D_all_2_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cpp ${ccflags} ${ldflags})
            add_custom_mpi_host_test(halo_exchange_3D_all_3_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cpp ${ccflags} ${ldflags})
            add_custom_mpi_host_test(halo_exchange_3D_generic_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cpp ${ccflags} ${ldflags})
            add_custom_mpi_host_test(halo_exchange_3D_generic_full_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cpp ${ccflags} ${ldflags})

            set(ccflags " ${HOST_SPECIFIC_OPTIONS}" )
            set(ldflags " ")

            add_custom_mpi_mic_test(halo_exchange_3D halo_exchange_3D.cpp ${ccflags} ${ldflags})

            add_custom_mpi_mic_test(all_to_all_halo_3D ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_all_to_all_halo_3D.cpp ${ccflags} ${ldflags})

            add_custom_mpi_mic_test(halo_exchange_3D_all ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cpp ${ccflags} ${ldflags})
            add_custom_mpi_mic_test(halo_exchange_3D_all_2 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cpp ${ccflags} ${ldflags})
            add_custom_mpi_mic_test(halo_exchange_3D_all_3 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cpp ${ccflags} ${ldflags})
            add_custom_mpi_mic_test(halo_exchange_3D_generic ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cpp ${ccflags} ${ldflags})
            add_custom_mpi_mic_test(halo_exchange_3D_generic_full ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cpp ${ccflags} ${ldflags})

            set(ccflags " ${HOST_SPECIFIC_OPTIONS} -DVECTOR_INTERFACE" )
            set(ldflags " ")

            add_custom_mpi_mic_test(halo_exchange_3D_all_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cpp ${ccflags} ${ldflags})
            add_custom_mpi_mic_test(halo_exchange_3D_all_2_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cpp ${ccflags} ${ldflags})
            add_custom_mpi_mic_test(halo_exchange_3D_all_3_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cpp ${ccflags} ${ldflags})
            add_custom_mpi_mic_test(halo_exchange_3D_generic_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cpp ${ccflags} ${ldflags})
            add_custom_mpi_mic_test(halo_exchange_3D_generic_full_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cpp ${ccflags} ${ldflags})
        endif()

        if ( NOT "${DISABLE_MPI_TESTS_ON_TARGET}" STREQUAL "GPU")
            set(ccflags " ")
            set(ldflags " ")

            add_custom_mpi_gpu_test(halo_exchange_3D_all ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(halo_exchange_3D_all_2 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(halo_exchange_3D_all_3 ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(halo_exchange_3D_generic ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(halo_exchange_3D_generic_full ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cu ${ccflags} ${ldflags})

            set(ccflags " -DVECTOR_INTERFACE")
            set(ldflags " ")

            add_custom_mpi_gpu_test(halo_exchange_3D_all_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(halo_exchange_3D_all_2_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_2.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(halo_exchange_3D_all_3_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_all_3.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(halo_exchange_3D_generic_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic.cu ${ccflags} ${ldflags})
            add_custom_mpi_gpu_test(halo_exchange_3D_generic_full_v ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/communication/test_halo_exchange_3D_generic_full.cu ${ccflags} ${ldflags})
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
