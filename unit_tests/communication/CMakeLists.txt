# collect all tests in current folder
fetch_x86_tests(. LABELS unittest_x86 )
fetch_mc_tests(. LABELS unittest_mc )
fetch_gpu_tests(. LABELS unittest_gpu )

add_custom_x86_test(
    TARGET boolist
    SOURCES boolist.cpp
    LABELS unittest_x86
    )
add_custom_x86_test(
    TARGET ndloops
    SOURCES ndloops.cpp
    LABELS unittest_x86
    )
add_custom_mc_test(
    TARGET boolist
    SOURCES boolist.cpp
    LABELS unittest_mc
    )
add_custom_mc_test(
    TARGET ndloops
    SOURCES ndloops.cpp
    LABELS unittest_mc
    )

# custom test cases
if( NOT GT_GCL_ONLY )
    if( GT_USE_MPI )
        set(testdir ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication)

        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D
            SOURCES halo_exchange_3D.cpp
            LABELS mpitest_x86
            )

        add_custom_mpi_x86_test(
            TARGET all_to_all_halo_3D
            SOURCES ${testdir}/test_all_to_all_halo_3D.cpp
            LABELS mpitest_x86
            )

        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_all
            SOURCES ${testdir}/test_halo_exchange_3D_all.cpp
            LABELS mpitest_x86
            )
        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_all_2
            SOURCES ${testdir}/test_halo_exchange_3D_all_2.cpp
            LABELS mpitest_x86
            )
        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_all_3
            SOURCES ${testdir}/test_halo_exchange_3D_all_3.cpp
            LABELS mpitest_x86
            )
        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_generic
            SOURCES ${testdir}/test_halo_exchange_3D_generic.cpp
            LABELS mpitest_x86
            )
        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_generic_full
            SOURCES ${testdir}/test_halo_exchange_3D_generic_full.cpp
            LABELS mpitest_x86
            )

        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_all_v
            SOURCES ${testdir}/test_halo_exchange_3D_all.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_x86
            )
        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_all_2_v
            SOURCES ${testdir}/test_halo_exchange_3D_all_2.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_x86
            )
        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_all_3_v
            SOURCES ${testdir}/test_halo_exchange_3D_all_3.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_x86
            )
        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_generic_v
            SOURCES ${testdir}/test_halo_exchange_3D_generic.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_x86
            )
        add_custom_mpi_x86_test(
            TARGET halo_exchange_3D_generic_full_v
            SOURCES ${testdir}/test_halo_exchange_3D_generic_full.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_x86
            )

        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D
            SOURCES halo_exchange_3D.cpp
            LABELS mpitest_mc
            )

        add_custom_mpi_mc_test(
            TARGET all_to_all_halo_3D
            SOURCES ${testdir}/test_all_to_all_halo_3D.cpp
            LABELS mpitest_mc
            )

        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_all
            SOURCES ${testdir}/test_halo_exchange_3D_all.cpp
            LABELS mpitest_mc
            )
        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_all_2
            SOURCES ${testdir}/test_halo_exchange_3D_all_2.cpp
            LABELS mpitest_mc
            )
        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_all_3
            SOURCES ${testdir}/test_halo_exchange_3D_all_3.cpp
            LABELS mpitest_mc
            )
        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_generic
            SOURCES ${testdir}/test_halo_exchange_3D_generic.cpp
            LABELS mpitest_mc
            )
        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_generic_full
            SOURCES ${testdir}/test_halo_exchange_3D_generic_full.cpp
            LABELS mpitest_mc
            )

        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_all_v
            SOURCES ${testdir}/test_halo_exchange_3D_all.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_mc
            )
        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_all_2_v
            SOURCES ${testdir}/test_halo_exchange_3D_all_2.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_mc
            )
        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_all_3_v
            SOURCES ${testdir}/test_halo_exchange_3D_all_3.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_mc
            )
        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_generic_v
            SOURCES ${testdir}/test_halo_exchange_3D_generic.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_mc
            )
        add_custom_mpi_mc_test(
            TARGET halo_exchange_3D_generic_full_v
            SOURCES ${testdir}/test_halo_exchange_3D_generic_full.cpp
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_mc
            )

        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_all
            SOURCES ${testdir}/test_halo_exchange_3D_all.cu
            LABELS mpitest_cuda
            )
        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_all_2
            SOURCES ${testdir}/test_halo_exchange_3D_all_2.cu
            LABELS mpitest_cuda
            )
        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_all_3
            SOURCES ${testdir}/test_halo_exchange_3D_all_3.cu
            LABELS mpitest_cuda
            )
        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_generic
            SOURCES ${testdir}/test_halo_exchange_3D_generic.cu
            LABELS mpitest_cuda
            )
        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_generic_full
            SOURCES ${testdir}/test_halo_exchange_3D_generic_full.cu
            LABELS mpitest_cuda
            )

        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_all_v
            SOURCES ${testdir}/test_halo_exchange_3D_all.cu
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_cuda
            )
        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_all_2_v
            SOURCES ${testdir}/test_halo_exchange_3D_all_2.cu
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_cuda
            )
        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_all_3_v
            SOURCES ${testdir}/test_halo_exchange_3D_all_3.cu
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_cuda
            )
        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_generic_v
            SOURCES ${testdir}/test_halo_exchange_3D_generic.cu
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_cuda
            )
        add_custom_mpi_gpu_test(
            TARGET halo_exchange_3D_generic_full_v
            SOURCES ${testdir}/test_halo_exchange_3D_generic_full.cu
            COMPILE_DEFINITIONS VECTOR_INTERFACE
            LABELS mpitest_cuda
            )

    endif()
endif()
