gridtools_add_unit_test(test_extents SOURCES test_extents.cpp)
if (CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    gridtools_add_unit_test(test_fn_deref SOURCES test_fn_deref.cpp)
    gridtools_add_unit_test(test_fn_offsets SOURCES test_fn_offsets.cpp)
    gridtools_add_unit_test(test_fn_shift SOURCES test_fn_shift.cpp)
endif()

if(TARGET _gridtools_cuda)
    gridtools_add_unit_test(test_extents_cuda
        SOURCES test_extents.cu
        LIBRARIES _gridtools_cuda
        LABELS cuda)
endif()
