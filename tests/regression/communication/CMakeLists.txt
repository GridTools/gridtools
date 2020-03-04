if(NOT TARGET MPI::MPI_CXX)
  return()
endif()

function(gridtools_add_communication_test tgt_name)
    foreach(arch IN LISTS GT_GCL_ARCHS)
        set(tgt ${tgt_name}_${arch})
        gridtools_add_test_executable(${tgt} ${ARGN} LIBRARIES GridToolsTest mpi_gtest_main gmock gcl_${arch})
        if(arch STREQUAL gpu)
            target_compile_definitions(${tgt} PRIVATE GT_STORAGE_CUDA GT_GCL_GPU)
        elseif(arch STREQUAL cpu)
            target_compile_definitions(${tgt} PRIVATE GT_STORAGE_X86 GT_GCL_CPU)
        endif()
        set(nproc 4)
        set(labels mpi gcl)
        # Note: We use MPITEST_ instead of MPIEXEC_ because our own MPI_TEST_-variables are slurm-aware
        add_test(
                NAME ${tgt}
                COMMAND  ${MPITEST_EXECUTABLE} ${MPITEST_NUMPROC_FLAG} ${nproc} ${MPITEST_PREFLAGS} $<TARGET_FILE:${tgt}> ${MPITEST_POSTFLAGS}
        )
        set_tests_properties(${tgt} PROPERTIES LABELS "${labels}")
        set_tests_properties(${tgt} PROPERTIES PROCESSORS ${nproc})
    endforeach()
endfunction()

function(gridtools_add_all_to_all_halo_3D_test)
    gridtools_add_test_executable(test_all_to_all_halo_3D
            SOURCES test_all_to_all_halo_3D.cpp
            LIBRARIES mpi_gtest_main)
    set(nproc 4)
    set(labels mpi gcl)
    # Note: We use MPITEST_ instead of MPIEXEC_ because our own MPI_TEST_-variables are slurm-aware
    add_test(
            NAME test_all_to_all_halo_3D
            COMMAND  ${MPITEST_EXECUTABLE} ${MPITEST_NUMPROC_FLAG} ${nproc} ${MPITEST_PREFLAGS} $<TARGET_FILE:test_all_to_all_halo_3D> ${MPITEST_POSTFLAGS}
    )
    set_tests_properties(test_all_to_all_halo_3D PROPERTIES LABELS "${labels}")
    set_tests_properties(test_all_to_all_halo_3D PROPERTIES PROCESSORS ${nproc})
endfunction()

gridtools_add_communication_test(test_halo_exchange_3D SOURCES test_halo_exchange_3D.cpp)
gridtools_add_all_to_all_halo_3D_test()