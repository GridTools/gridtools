
if (OpenMP_CXX_FOUND)
    set(omp_ OpenMP::OpenMP_CXX)
endif()
gridtools_add_test_library(regression_main SOURCES regression_fixture.cpp  LIBRARIES gtest PRIVATE ${omp_})
unset(omp_)

foreach(arch IN LISTS GT_GCL_ARCHS)
    gridtools_add_test_library(mpi_gtest_main_${arch}
            SOURCES mpi_test_driver.cpp
            LIBRARIES gtest gcl_${arch})
endforeach()
