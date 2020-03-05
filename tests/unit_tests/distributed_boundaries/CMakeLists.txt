gridtools_add_unit_test(test_bindbc_utilities SOURCES test_bindbc_utilities.cpp)

if (gcl_cpu)
    gridtools_add_mpi_test(cpu test_distributed_boundaries_cpu SOURCES test_distributed_boundaries.cpp)
    target_compile_definitions(test_distributed_boundaries_cpu PRIVATE GT_STORAGE_X86 GT_GCL_CPU GT_TIMER_OMP)
endif()

if (gcl_gpu)
    gridtools_add_mpi_test(gpu test_distributed_boundaries_gpu SOURCES test_distributed_boundaries.cpp)
    target_compile_definitions(test_distributed_boundaries_gpu PRIVATE GT_STORAGE_CUDA GT_GCL_GPU GT_TIMER_CUDA)
endif()
