# collect all tests in current folder
fetch_x86_tests(.)
fetch_mc_tests(.)
fetch_gpu_tests(.)


add_custom_x86_test( TARGET boolist SOURCES boolist.cpp)
add_custom_x86_test( TARGET ndloops SOURCES ndloops.cpp)
add_custom_mc_test( TARGET boolist SOURCES boolist.cpp)
add_custom_mc_test( TARGET ndloops SOURCES ndloops.cpp)

# custom test cases
if( NOT GCL_ONLY )
    if( USE_MPI )
        set(ccflags ${X86_SPECIFIC_OPTIONS} )

        if( NOT "${DISABLE_MPI_TESTS_ON_TARGET}" STREQUAL "CPU")
            add_custom_mpi_x86_test(TARGET halo_exchange_3D SOURCES halo_exchange_3D.cpp ADDITIONAL_FLAGS ${ccflags})

            add_custom_mpi_x86_test(TARGET all_to_all_halo_3D SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_all_to_all_halo_3D.cpp ADDITIONAL_FLAGS ${ccflags})

            add_custom_mpi_x86_test(TARGET halo_exchange_3D_all SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_x86_test(TARGET halo_exchange_3D_all_2 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_2.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_x86_test(TARGET halo_exchange_3D_all_3 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_3.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_x86_test(TARGET halo_exchange_3D_generic SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_x86_test(TARGET halo_exchange_3D_generic_full SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic_full.cpp ADDITIONAL_FLAGS ${ccflags})

            set(ccflags " ${X86_SPECIFIC_OPTIONS} -DVECTOR_INTERFACE" )

            add_custom_mpi_x86_test(TARGET halo_exchange_3D_all_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_x86_test(TARGET halo_exchange_3D_all_2_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_2.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_x86_test(TARGET halo_exchange_3D_all_3_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_3.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_x86_test(TARGET halo_exchange_3D_generic_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_x86_test(TARGET halo_exchange_3D_generic_full_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic_full.cpp ADDITIONAL_FLAGS ${ccflags})

            set(ccflags " ${X86_SPECIFIC_OPTIONS}" )

            add_custom_mpi_mc_test(TARGET halo_exchange_3D SOURCES halo_exchange_3D.cpp ADDITIONAL_FLAGS ${ccflags})

            add_custom_mpi_mc_test(TARGET all_to_all_halo_3D SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_all_to_all_halo_3D.cpp ADDITIONAL_FLAGS ${ccflags})

            add_custom_mpi_mc_test(TARGET halo_exchange_3D_all SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_mc_test(TARGET halo_exchange_3D_all_2 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_2.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_mc_test(TARGET halo_exchange_3D_all_3 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_3.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_mc_test(TARGET halo_exchange_3D_generic SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_mc_test(TARGET halo_exchange_3D_generic_full SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic_full.cpp ADDITIONAL_FLAGS ${ccflags})

            set(ccflags " ${X86_SPECIFIC_OPTIONS} -DVECTOR_INTERFACE" )

            add_custom_mpi_mc_test(TARGET halo_exchange_3D_all_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_mc_test(TARGET halo_exchange_3D_all_2_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_2.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_mc_test(TARGET halo_exchange_3D_all_3_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_3.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_mc_test(TARGET halo_exchange_3D_generic_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic.cpp ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_mc_test(TARGET halo_exchange_3D_generic_full_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic_full.cpp ADDITIONAL_FLAGS ${ccflags})
        endif()

        if ( NOT "${DISABLE_MPI_TESTS_ON_TARGET}" STREQUAL "GPU")

            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_all SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all.cu )
            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_all_2 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_2.cu )
            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_all_3 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_3.cu )
            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_generic SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic.cu )
            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_generic_full SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic_full.cu )

            set(ccflags -DVECTOR_INTERFACE)

            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_all_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all.cu ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_all_2_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_2.cu ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_all_3_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_all_3.cu ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_generic_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic.cu ADDITIONAL_FLAGS ${ccflags})
            add_custom_mpi_gpu_test(TARGET halo_exchange_3D_generic_full_v SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication/test_halo_exchange_3D_generic_full.cu ADDITIONAL_FLAGS ${ccflags})
        endif()

        endif()
endif()
