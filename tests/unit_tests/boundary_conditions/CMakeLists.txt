if(TARGET bc_gpu)
    gridtools_add_unit_test(test_apply_gpu_internals SOURCES test_apply_gpu_internals.cu LIBRARIES bc_gpu)
    gridtools_add_unit_test(test_boundary_conditions_gpu SOURCES test_boundary_conditions.cpp LIBRARIES bc_gpu)
    target_compile_definitions(test_boundary_conditions_gpu PRIVATE GT_STORAGE_CUDA GT_GCL_GPU)
endif()

if(TARGET bc_cpu)
    gridtools_add_unit_test(test_boundary_conditions_cpu SOURCES test_boundary_conditions.cpp LIBRARIES bc_cpu)
    target_compile_definitions(test_boundary_conditions_cpu PRIVATE GT_STORAGE_X86 GT_GCL_CPU)
endif()

