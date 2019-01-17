include (CMakeDependentOption)

option( GT_ENABLE_PYUTILS "If on, Python utility scripts will be configured" OFF)
option( GT_ENABLE_TARGET_CUDA "Compile CUDA GPU backend examples and unit tests" ${CUDA_AVAILABLE})
option( GT_ENABLE_TARGET_X86 "Compile x86 backend examples and unit tests" ${OPENMP_AVAILABLE} )
option( GT_ENABLE_TARGET_MC "Compile MC backend examples and unit tests" ${OPENMP_AVAILABLE} )
option( GT_USE_MPI "Compile with MPI support" ${MPI_AVAILABLE} )

# TODO remove when implementing smaller-grained test enablers
option( GT_GCL_ONLY "If on only library is build but not the examples and tests" OFF )

option( GT_TESTS_STRUCTURED_GRID "compile for rectangular grids" ON )

CMAKE_DEPENDENT_OPTION(
    GT_CUDA_PTX_GENERATION "Compile regression tests to intermediate representation"
    OFF "BUILD_TESTING" OFF)
CMAKE_DEPENDENT_OPTION(
    GT_ENABLE_PERFORMANCE_METERS "If on, meters will be reported for each stencil"
    OFF "BUILD_TESTING" OFF)
CMAKE_DEPENDENT_OPTION(
    GT_SINGLE_PRECISION "Option determining number of bytes used to represent the floating poit types (see defs.hpp for configuration)"
    OFF "BUILD_TESTING" OFF)
CMAKE_DEPENDENT_OPTION(
    GT_TESTS_ENABLE_CACHING "Enable caches in stencil composition for tests"
    ON "BUILD_TESTING" ON)
CMAKE_DEPENDENT_OPTION(
    GT_TREAT_WARNINGS_AS_ERROR "Treat warnings as errors"
    OFF "BUILD_TESTING" OFF)
set( GT_CXX_STANDARD "c++11" CACHE STRING "C++ standard to be used for compilation" )
set_property(CACHE GT_CXX_STANDARD PROPERTY STRINGS "c++11;c++14;c++17")

option( GT_ENABLE_EXPERIMENTAL_REPOSITORY "Enables downloading the gridtools_experimental repository" OFF )


option( GT_COMPILE_EXAMPLES "Specify examples should be compiled" ON )

CMAKE_DEPENDENT_OPTION(GT_INSTALL_EXAMPLES "Specify the path where to install sources and binaries of the examples" OFF "GT_COMPILE_EXAMPLES" OFF)

set(GT_INSTALL_EXAMPLES_PATH STRING "Specifies where the source codes and binary of examples should be installed"
    "${CMAKE_INSTALL_PREFIX}")

if (DEFINED ENV{CUDA_ARCH})
    set(GT_CUDA_ARCH_INIT $ENV{CUDA_ARCH})
else()
    set(GT_CUDA_ARCH_INIT "sm_35")
endif()
set(GT_CUDA_ARCH "${GT_CUDA_ARCH_INIT}" CACHE STRING "Compute capability for CUDA used for tests")

set( TEST_SCRIPT ${CMAKE_BINARY_DIR}/run_tests.sh )
set( TEST_MANIFEST ${CMAKE_BINARY_DIR}/tests_manifest.txt )
set( TEST_MPI_SCRIPT ${CMAKE_BINARY_DIR}/run_mpi_tests.sh )
set( TEST_CUDA_MPI_SCRIPT ${CMAKE_BINARY_DIR}/run_cuda_mpi_tests.sh )

mark_as_advanced(
    GT_CXX_STANDARD
    GT_TREAT_WARNINGS_AS_ERROR
    GT_CUDA_PTX_GENERATION
    )
