if (CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    gridtools_add_unit_test(test_fn_deref SOURCES test_fn_deref.cpp)
    gridtools_add_unit_test(test_fn_offsets SOURCES test_fn_offsets.cpp)
    gridtools_add_unit_test(test_fn_shift SOURCES test_fn_shift.cpp)
endif()

gridtools_add_unit_test(test_fn_scan SOURCES test_fn_scan.cpp)

if(TARGET _gridtools_cuda)
    gridtools_add_unit_test(test_fn_scan_cuda
            SOURCES test_fn_scan.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
endif()
