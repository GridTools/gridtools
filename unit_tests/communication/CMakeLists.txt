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

set(testdir ${CMAKE_CURRENT_SOURCE_DIR}/../../regression/communication)

set(SOURCES
    ${testdir}/test_halo_exchange_3D_all.cpp
    ${testdir}/test_halo_exchange_3D_all_2.cpp
    ${testdir}/test_halo_exchange_3D_all_3.cpp
    ${testdir}/test_halo_exchange_3D_generic.cpp
    ${testdir}/test_halo_exchange_3D_generic_full.cpp
    )
set(ADDITIONAL_SOURCES
    halo_exchange_3D.cpp
    ${testdir}/test_all_to_all_halo_3D.cpp
    )

# custom test cases
if( NOT GT_GCL_ONLY )
    if( GT_USE_MPI )
        if( NOT "${GT_DISABLE_MPI_TESTS_ON_TARGET}" STREQUAL "CPU")
            foreach (source IN LISTS ${SOURCES} ${ADDITIONAL_SOURCES})
                get_filename_component(target ${source})
                add_custom_mpi_x86_test(
                    TARGET ${target}
                    NPROC 4
                    SOURCES ${source}
                    LABELS mpitest_x86
                    )
                add_custom_mpi_mc_test(
                    TARGET ${target}
                    NPROC 4
                    SOURCES ${source}
                    LABELS mpitest_mc
                    )
            endforeach()
            foreach (source IN LISTS ${SOURCES})
                get_filename_component(target ${source})
                add_custom_mpi_x86_test(
                    TARGET ${target}
                    NPROC 4
                    SOURCES ${source}
                    COMPILE_DEFINITIONS VECTOR_INTERFACE
                    LABELS mpitest_x86
                    )
                add_custom_mpi_mc_test(
                    TARGET ${target}
                    NPROC 4
                    SOURCES ${source}
                    COMPILE_DEFINITIONS VECTOR_INTERFACE
                    LABELS mpitest_mc
                    )
            endforeach()
        endif()

        if ( NOT "${GT_DISABLE_MPI_TESTS_ON_TARGET}" STREQUAL "GPU")
            foreach (source IN LISTS ${SOURCES})
                get_filename_component(target ${source})
                add_custom_mpi_gpu_test(
                    TARGET ${target}
                    NPROC 4
                    SOURCES ${source}
                    LABELS mpitest_cuda
                    )
                add_custom_mpi_gpu_test(
                    TARGET ${target}
                    NPROC 4
                    SOURCES ${source}
                    COMPILE_DEFINITIONS VECTOR_INTERFACE
                    LABELS mpitest_cuda
                    )
            endforeach()
        endif()
    endif()
endif()
