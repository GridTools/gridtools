set( GT_BOOST_FUSION_MAX_SIZE 20 CACHE STRING "max sizes of boost fusion containers" ) # -> not CACHE (only test, regression)
option( GT_ENABLE_PERFORMANCE_METERS "If on, meters will be reported for each stencil" OFF) # -> regression
option( GT_ENABLE_PYUTILS "If on, Python utility scripts will be configured" OFF)
option( GT_ENABLE_TARGET_CUDA "Compile CUDA GPU backend examples and unit tests" ${CUDA_AVAILABLE})
option( GT_ENABLE_TARGET_X86 "Compile x86 backend examples and unit tests" ON )
option( GT_ENABLE_TARGET_MC "Compile MC backend examples and unit tests" ON )
option( GT_USE_MPI "Compile with MPI support" ${MPI_AVAILABLE} )
option( GT_GCL_ONLY "If on only library is build but not the examples and tests" OFF ) # -> GCL component

option( GT_CUDA_PTX_GENERATION "Compile to intermediate representation" OFF ) # -> only enabled if disabled_testing = OFF
option( SINGLE_PRECISION "Option determining number of bytes used to represent the floating poit types (see defs.hpp for configuration)" OFF ) # -> test, regression
option( STRUCTURED_GRIDS "compile for rectangular grids" ON ) # -> property
option( GT_TESTS_ENABLE_CACHING "Enable caching functionality" ON)
option( GT_TREAT_WARNINGS_AS_ERROR "Treat warnings as errors" OFF )
set( GT_CXX_STANDARD "c++11" CACHE STRING "C++ standard to be used for compilation" )
set_property(CACHE GT_CXX_STANDARD PROPERTY STRINGS "c++11;c++14;c++17")
option( COMPILE_EXAMPLES "Compiles the codes in examples folder" ON ) # -> remove (GT_INSTALL_EXAMPLES)
option( GT_DISABLE_MPI_TESTS_ON_TARGET "Disables all the cpu communication tests" OFF )
set_property(CACHE GT_DISABLE_MPI_TESTS_ON_TARGET PROPERTY STRINGS OFF CPU GPU)
option( ENABLE_EXPERIMENTAL_REPOSITORY "Enables downloading the gridtools_experimental repository" OFF )
option( GT_INSTALL_EXAMPLES "Specify if source codes and binaries of examples should be installed somewhere" OFF )
set( GT_INSTALL_EXAMPLES_PATH "${CMAKE_INSTALL_PREFIX}" CACHE STRING "Specify where the source codes and binaries of examples should be installed" )
if (GT_ENABLE_TARGET_CUDA)
    if (DEFINED ENV{CUDA_ARCH})
        set(GT_CUDA_ARCH_INIT $ENV{CUDA_ARCH})
    else()
        set(GT_CUDA_ARCH_INIT "sm_35")
    endif()
    set( GT_CUDA_ARCH ${GT_CUDA_ARCH_INIT} CACHE STRING "Compute capability for CUDA" )
endif()

set( GCL_MPI "${GT_USE_MPI}" )
set( GCL_GPU "${GT_ENABLE_TARGET_CUDA}" )
set( TEST_SCRIPT ${CMAKE_BINARY_DIR}/run_tests.sh )
set( TEST_MANIFEST ${CMAKE_BINARY_DIR}/tests_manifest.txt )
set( TEST_MPI_SCRIPT ${CMAKE_BINARY_DIR}/run_mpi_tests.sh )
set( TEST_CUDA_MPI_SCRIPT ${CMAKE_BINARY_DIR}/run_cuda_mpi_tests.sh )

mark_as_advanced(
    GT_CXX_STANDARD
    GT_TREAT_WARNINGS_AS_ERROR
    GT_CUDA_PTX_GENERATION
    )
