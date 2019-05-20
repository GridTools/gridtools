include (CMakeDependentOption)

set( CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON CACHE INTERNAL "" )

option( GT_ENABLE_PYUTILS "If on, Python utility scripts will be configured" OFF)
option( GT_ENABLE_BACKEND_CUDA "Compile CUDA GPU backend examples and unit tests" ${CUDA_AVAILABLE})
option( GT_ENABLE_BACKEND_X86 "Compile x86 backend examples and unit tests" ${OPENMP_AVAILABLE} )
option( GT_ENABLE_BACKEND_NAIVE "Compile naive backend examples and unit tests" ON)
option( GT_ENABLE_BACKEND_MC "Compile MC backend examples and unit tests" ${OPENMP_AVAILABLE} )
option( GT_USE_MPI "Compile with MPI support" ${MPI_AVAILABLE} )

# TODO remove when implementing smaller-grained test enablers
option( GT_GCL_ONLY "If on only library is build but not the examples and tests" OFF )

option( GT_TESTS_ICOSAHEDRAL_GRID "compile tests for icosahedral grids" OFF )

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
set( GT_CXX_STANDARD "c++14" CACHE STRING "C++ standard to be used for compilation" )
set_property(CACHE GT_CXX_STANDARD PROPERTY STRINGS "c++14;c++17")

option( GT_ENABLE_EXPERIMENTAL_REPOSITORY "Enables downloading the gridtools_experimental repository" OFF )

#if we are pointing to the default install path (usually system) we will disable installation of examples by default    
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(_default_GT_INSTALL_EXAMPLES OFF)
else()
    set(_default_GT_INSTALL_EXAMPLES ON)
endif()
option(GT_INSTALL_EXAMPLES "Install example sources" ${_default_GT_INSTALL_EXAMPLES})
if(GT_INSTALL_EXAMPLES)
    set(GT_INSTALL_EXAMPLES_PATH "${CMAKE_INSTALL_PREFIX}/gridtools_examples" CACHE FILEPATH 
        "Specifies where the source codes of examples should be installed")
    mark_as_advanced(CLEAR GT_INSTALL_EXAMPLES_PATH)
else()
    if(GT_INSTALL_EXAMPLES_PATH)
        mark_as_advanced(FORCE GT_INSTALL_EXAMPLES_PATH)
    endif()
endif()

if (DEFINED ENV{CUDA_ARCH})
    set(GT_CUDA_ARCH_INIT $ENV{CUDA_ARCH})
else()
    set(GT_CUDA_ARCH_INIT "sm_35")
endif()
set(GT_CUDA_ARCH "${GT_CUDA_ARCH_INIT}" CACHE STRING "Compute capability for CUDA used for tests")

mark_as_advanced(
    GT_CXX_STANDARD
    GT_TREAT_WARNINGS_AS_ERROR
    GT_CUDA_PTX_GENERATION
    )
