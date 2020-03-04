gridtools_add_unit_test(test_bindbc_utilities SOURCES test_bindbc_utilities.cpp)

if(NOT TARGET MPI::MPI_CXX)
    return()
endif()

function(gridtools_add_distributed_boundaries_test tgt_name)
    foreach(arch IN LISTS GT_GCL_ARCHS)
        set(tgt ${tgt_name}_${arch})
        gridtools_add_test_executable(${tgt} ${ARGN} LIBRARIES GridToolsTest mpi_gtest_main gmock gcl_${arch})
        if(arch STREQUAL gpu)
            target_compile_definitions(${tgt} PRIVATE GT_STORAGE_CUDA GT_GCL_GPU GT_TIMER_CUDA)
        elseif(arch STREQUAL cpu)
            target_compile_definitions(${tgt} PRIVATE GT_STORAGE_X86 GT_GCL_CPU GT_TIMER_OMP)
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

gridtools_add_distributed_boundaries_test(test_distributed_boundaries SOURCES test_distributed_boundaries.cpp)
