if(TARGET gridtools_cuda)
    gridtools_add_unit_test(test_layout_transformation_cuda
            SOURCES test_layout_transformation.cpp
            LIBRARIES gridtools_cuda
            LABELS cuda)
endif()

if(OpenMP_CXX_FOUND)
    gridtools_add_unit_test(test_layout_transformation
            SOURCES test_layout_transformation.cpp
            LIBRARIES OpenMP::OpenMP_CXX
            NO_NVCC)
else()
    gridtools_add_unit_test(test_layout_transformation
            SOURCES test_layout_transformation.cpp
            NO_NVCC)
endif()
