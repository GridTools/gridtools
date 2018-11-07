cmake_policy(SET CMP0048 NEW)
project(GridTools VERSION "0.1")

cmake_minimum_required(VERSION 3.0)
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ===============
# Load some cmake modules.
# 1) define what flags should be available,

include (flags)

if (ENABLE_CUDA)
    enable_language (CXX Fortran CUDA C)
else()
    enable_language (CXX Fortran C)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

if( GRIDTOOLS_MODULES )
# To be completed!

  add_subdirectory( modules/storage )

else()

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
include (gt_bindings)
include (components)

if( COMPONENT_COMMON )
    generate_target_for(NAME common)
    generate_install_targets_for(common common)
endif()

if( COMPONENT_BOUNDARY_CONDITIONS )
    generate_target_for(NAME boundary-conditions)
    generate_install_targets_for(boundary-conditions boundary-conditions)
endif()

if( COMPONENT_STENCIL_COMPOSITION )
    generate_target_for(NAME stencil-composition)
    generate_install_targets_for(stencil-composition stencil-composition)
endif()

if( COMPONENT_GCL )
    generate_target_for(NAME gcl SOURCES src/GCL.cpp include/gridtools/communication/GCL.hpp include/gridtools/communication/high-level/stats_collector.hpp )
    generate_install_targets_for(gcl communication)
    target_compile_options(gcl PUBLIC ${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS} PRIVATE -std=c++11)
    set(exe_LIBS ${exe_LIBS} gcl)
endif()

if( COMPONENT_DISTRIBUTED_BOUNDARIES )
    generate_target_for(NAME distributed-boundaries)
    generate_install_targets_for(distributed-boundaries distributed-boundaries)
endif()

if( COMPONENT_STORAGE )
    generate_target_for(NAME storage)
    generate_install_targets_for(storage storage)
endif()

if( COMPONENT_INTERFACE )
    generate_target_for(NAME interface)
    generate_install_targets_for(interface interface)
endif()

if( COMPONENT_TOOLS )
    generate_target_for(NAME tools)
    generate_install_targets_for(tools tools)
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
if(ENABLE_PYUTILS)
    add_subdirectory(pyutils)
endif()

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE GridTools)


configure_file(GridToolsConfig.cmake.in
  "${PROJECT_BINARY_DIR}/cmake/GridToolsConfig.cmake" @ONLY)
configure_file(GridToolsConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/cmake/GridToolsConfigVersion.cmake" @ONLY)

# Install the GridToolsConfig.cmake and GridToolsConfigVersion.cmake
install(FILES "${PROJECT_BINARY_DIR}/cmake/GridToolsConfig.cmake"
  "${PROJECT_BINARY_DIR}/cmake/GridToolsConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT dev)

# TODO for GC-546: move this to an appropriate location
if(COMPONENT_C_BINDINGS)
    generate_target_for(NAME c_bindings)
    generate_install_targets_for(c_bindings c_bindings)
    install(FILES
        "${PROJECT_SOURCE_DIR}/cmake/gt_bindings.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/gt_bindings_generate.cmake"
        DESTINATION "cmake")
    install(FILES
        "${PROJECT_SOURCE_DIR}/src/c_bindings/generator.cpp"
        "${PROJECT_SOURCE_DIR}/src/c_bindings/generator_main.cpp"
        "${PROJECT_SOURCE_DIR}/src/c_bindings/array_descriptor.f90"
        "${PROJECT_SOURCE_DIR}/src/c_bindings/handle.f90"
        "${PROJECT_SOURCE_DIR}/src/c_bindings/handle.cpp"
        DESTINATION "src/c_bindings")
endif()

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

endif()
