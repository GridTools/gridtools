# collect all tests in current folder
fetch_x86_tests(. LABELS unittest_x86 )
fetch_mc_tests(. LABELS unittest_mc )
fetch_gpu_tests(. LABELS unittest_gpu )

add_custom_test(
    x86
    TARGET boolist
    SOURCES boolist.cpp
    LABELS unittest_x86
    )
add_custom_test(
    x86
    TARGET ndloops
    SOURCES ndloops.cpp
    LABELS unittest_x86
    )
add_custom_test(
    mc
    TARGET boolist
    SOURCES boolist.cpp
    LABELS unittest_mc
    )
add_custom_test(
    mc
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
        foreach (source IN LISTS SOURCES ADDITIONAL_SOURCES)
            get_filename_component(target ${source} NAME_WE )
            add_custom_mpi_test(
                x86
                TARGET ${target}
                NPROC 4
                SOURCES ${source}
                LABELS mpitest_x86
                )
            add_custom_mpi_test(
                mc
                TARGET ${target}
                NPROC 4
                SOURCES ${source}
                LABELS mpitest_mc
                )
        endforeach()
        foreach (source IN LISTS SOURCES)
            get_filename_component(target ${source} NAME_WE )

            add_custom_mpi_test(
                x86
                TARGET ${target}_vector
                NPROC 4
                SOURCES ${source}
                COMPILE_DEFINITIONS VECTOR_INTERFACE
                LABELS mpitest_x86
                )
            add_custom_mpi_test(
                mc
                TARGET ${target}_vector
                NPROC 4
                SOURCES ${source}
                COMPILE_DEFINITIONS VECTOR_INTERFACE
                LABELS mpitest_mc
                )
        endforeach()

        foreach (source IN LISTS SOURCES)
            get_filename_component(name ${source} NAME )
            get_filename_component(path ${source} DIRECTORY )
            get_filename_component(source_name ${name} NAME_WE )
            set (source_ ${path}/${source_name}.cu)
            set(target ${source_name})

            add_custom_mpi_test(
                cuda
                TARGET ${target}
                NPROC 4
                SOURCES ${source_}
                LABELS mpitest_cuda
                )
            add_custom_mpi_test(
                cuda
                TARGET ${target}_vector
                NPROC 4
                SOURCES ${source_}
                COMPILE_DEFINITIONS VECTOR_INTERFACE
                LABELS mpitest_cuda
                )
        endforeach()
    endif()
endif()
