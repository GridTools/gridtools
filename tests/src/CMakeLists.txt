if (OpenMP_CXX_FOUND)
    gridtools_add_test_library(regression_main
            SOURCES regression_fixture.cpp
            LIBRARIES gtest OpenMP::OpenMP_CXX)
else()
    gridtools_add_test_library(regression_main
            SOURCES regression_fixture.cpp
            LIBRARIES gtest)
endif()

if(MPI_FOUND)
    if(TARGET gridtools_cuda)
        gridtools_add_test_library(mpi_gtest_main
                SOURCES mpi_test_driver.cpp
                LIBRARIES gtest MPI::MPI_CXX gridtools_cuda)
    else()
        gridtools_add_test_library(mpi_gtest_main
                SOURCES mpi_test_driver.cpp
                LIBRARIES gtest MPI::MPI_CXX)
    endif()
endif()
