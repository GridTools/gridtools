gridtools_add_unit_test(test_extents SOURCES test_extents.cpp)
gridtools_add_unit_test(test_fn_backend_naive SOURCES test_fn_backend_naive.cpp)
gridtools_add_unit_test(test_fn_scan SOURCES test_fn_scan.cpp)
if (CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    gridtools_add_unit_test(test_fn_deref SOURCES test_fn_deref.cpp)
    gridtools_add_unit_test(test_fn_offsets SOURCES test_fn_offsets.cpp)
    gridtools_add_unit_test(test_fn_shift SOURCES test_fn_shift.cpp)
endif()

if(TARGET _gridtools_cuda)
    gridtools_add_unit_test(test_fn_backend_gpu_cuda
            SOURCES test_fn_backend_gpu.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
    gridtools_add_unit_test(test_fn_scan_cuda
            SOURCES test_fn_scan.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
    gridtools_add_unit_test(test_extents_cuda
        SOURCES test_extents.cu
        LIBRARIES _gridtools_cuda
        LABELS cuda)
endif()
