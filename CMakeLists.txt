cmake_policy(SET CMP0048 NEW)
project(GridTools VERSION "0.1")

cmake_minimum_required(VERSION 3.12.4)
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ===============
# Load some cmake modules.
# 1) define what flags should be available,

enable_language (CXX Fortran C)
if (NOT DEFINED GT_ENABLE_TARGET_CUDA)
    # detect CUDA support
    include(CheckLanguage)
    check_language (CUDA)

    if (CMAKE_CUDA_COMPILER)
        enable_language (CUDA)
        message (STATUS "CUDA enabled")
        set (CUDA_AVAILABLE ON)
    else ()
        message (STATUS "CUDA disabled")
        set (CUDA_AVAILABLE OFF)
    endif ()
elseif (GT_ENABLE_TARGET_CUDA)
    enable_language (CUDA)
    message (STATUS "CUDA enabled")
    set (CUDA_AVAILABLE ON)
else ()
    message (STATUS "CUDA disabled")
    set (CUDA_AVAILABLE OFF)
endif ()

if (NOT DEFINED GT_USE_MPI)
    find_package(MPI COMPONENTS CXX)
    if (MPI_FOUND)
        message (STATUS "MPI enabled")
        set (MPI_AVAILABLE ON)
    else ()
        message (STATUS "MPI disabled")
        set (MPI_AVAILABLE OFF)
    endif()
elseif (GT_USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    message (STATUS "MPI enabled")
    set (MPI_AVAILABLE ON)
else ()
    message (STATUS "MPI disabled")
    set (MPI_AVAILABLE OFF)
endif()

include (flags)

if(POLICY CMP0060)
  # Policy to avoid cmake to substitute libraries with paths and extensions with -l<libname>
  cmake_policy(SET CMP0060 NEW)
endif()

# ===============
# Load some cmake modules.
# 2) package by name finder
# 3) testing methods
# 4) build configurator (include path, compiler flags, etc.
# ===============
include (definitions)
include (testing)
include (bindings)
include (components)

if( COMPONENT_COMMON )
    find_package( Boost 1.58 REQUIRED )
    find_package( OpenMP REQUIRED )
    find_package( Threads REQUIRED )

    generate_target_for(NAME common)
    generate_install_targets_for(common common)
    target_link_libraries( common INTERFACE Boost::boost OpenMP::OpenMP_CXX Threads::Threads)
    target_compile_options( common INTERFACE ${GT_CXX_MANDATORY_FLAGS} )
    if (GT_ENABLE_TARGET_CUDA)
        target_compile_options( common INTERFACE ${GPU_SPECIFIC_FLAGS} ${GT_CUDA_MANDATORY_FLAGS} )
        target_include_directories( common INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
        target_link_libraries( common INTERFACE ${CUDA_CUDART_LIBRARY} )
    endif()

endif()

if( COMPONENT_BOUNDARY_CONDITIONS )
    generate_target_for(NAME boundary-conditions)
    generate_install_targets_for(boundary-conditions boundary-conditions)
    target_link_libraries( boundary-conditions INTERFACE common )
endif()

if( COMPONENT_STENCIL_COMPOSITION )
    generate_target_for(NAME stencil-composition)
    generate_install_targets_for(stencil-composition stencil-composition)
    target_link_libraries( stencil-composition INTERFACE common )
endif()

if( COMPONENT_GCL )
    generate_target_for(NAME gcl SOURCES src/GCL.cpp include/gridtools/communication/GCL.hpp include/gridtools/communication/high-level/stats_collector.hpp )
    generate_install_targets_for(gcl communication)
    target_link_libraries(gcl common)
    if( GT_USE_MPI )
        target_link_libraries(gcl MPI::MPI_CXX)
    endif()
endif()

if( COMPONENT_DISTRIBUTED_BOUNDARIES )
    generate_target_for(NAME distributed-boundaries)
    generate_install_targets_for(distributed-boundaries distributed-boundaries)
    target_link_libraries( distributed-boundaries INTERFACE boundary-conditions gcl )
endif()

if( COMPONENT_STORAGE )
    generate_target_for(NAME storage)
    generate_install_targets_for(storage storage)
    target_link_libraries( distributed-boundaries INTERFACE common )
endif()

# ===============
# compile c_bindings_generator:
# ===============

if( COMPONENT_C_BINDINGS )
    generate_target_for(NAME c_bindings_handle SOURCES src/c_bindings/handle.cpp)
    target_compile_options(c_bindings_handle PRIVATE ${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS})
    # set_target_properties(c_bindings_handle PROPERTIES COMPILE_FLAGS "${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")
    generate_install_targets_for(c_bindings_handle c_bindings None)
    target_link_libraries(c_bindings_handle common)

    set(CMAKE_FORTRAN_MODOUT_FLAG ON)
    set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/mod)

    add_library(c_bindings_handle_fortran src/c_bindings/handle.f90)
    target_link_libraries(c_bindings_handle_fortran c_bindings_handle)
    target_include_directories(c_bindings_handle_fortran PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

    generate_target_for(NAME c_bindings_generator SOURCES src/c_bindings/generator.cpp)
    target_compile_options(c_bindings_generator PRIVATE ${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS})
    #set_target_properties(c_bindings_generator PROPERTIES COMPILE_FLAGS "${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")
    generate_install_targets_for(c_bindings_generator c_bindings None)
    target_link_libraries(c_bindings_generator common)

    add_library(c_bindings_generator_main src/c_bindings/generator_main.cpp)
    target_compile_options(c_bindings_generator_main PRIVATE ${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS})
    target_link_libraries(c_bindings_generator_main c_bindings_generator)
    generate_install_targets_for(c_bindings_generator_main c_bindings c_bindings_generator)

    add_library(array_descriptor src/c_bindings/array_descriptor.f90)
    generate_install_targets_for(array_descriptor c_bindings None)

    install(DIRECTORY
        "${PROJECT_BINARY_DIR}/mod/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gridtools")
    install(FILES
        "${PROJECT_SOURCE_DIR}/cmake/bindings.cmake"
        DESTINATION "${CMAKE_INSTALL_PREFIX}")

endif()

# ===============
# examples and tests
# ===============
if( INSTALL_GT_EXAMPLES )
   SET(COMPILE_EXAMPLES ON)
endif()

if( COMPILE_EXAMPLES )
  add_subdirectory(gt_examples)
endif()

include (fix_test_case_name)
if(NOT DISABLE_TESTING)
  add_subdirectory(regression)
  add_subdirectory(unit_tests)
endif()

# ===============
# Downloads the gridtools_experimental repository
# ===============
if(ENABLE_EXPERIMENTAL_REPOSITORY)
  include(gridtools_experimental/download_gridtools_experimental)
endif()

# ===============
# Python scripts
# ===============
if(GT_ENABLE_PYUTILS)
    add_subdirectory(pyutils)
endif()

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE GridTools)

include(CMakePackageConfigHelpers)
configure_package_config_file(GridToolsConfig.cmake.in
  ${PROJECT_BINARY_DIR}/GridToolsConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/GridToolsConfigVersion.cmake
  VERSION 1.2.3
  COMPATIBILITY SameMajorVersion )

# Install the GridToolsConfig.cmake and GridToolsConfigVersion.cmake
install(FILES "${PROJECT_BINARY_DIR}/GridToolsConfig.cmake"
  "${PROJECT_BINARY_DIR}/GridToolsConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake" COMPONENT dev)

if ( INSTALL_GT_EXAMPLES )
    install(DIRECTORY
            gt_examples
            DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/src"
            FILES_MATCHING REGEX ".*cpp|.*hpp")
endif()

# ===========
# Apply final actions
# ===========
include (post_config_actions)
