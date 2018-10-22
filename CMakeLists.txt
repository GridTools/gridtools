cmake_policy(SET CMP0048 NEW)
project(GridTools VERSION "0.1")
enable_language (CXX Fortran)

enable_language (CXX Fortran)

if( GRIDTOOLS_MODULES )
# To be completed!

  add_subdirectory( modules/storage )

else()

cmake_minimum_required(VERSION 3.0)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(POLICY CMP0060)
  # Policy to avoid cmake to substitute libraries with paths and extensions with -l<libname>
  cmake_policy(SET CMP0060 NEW)
endif()

# ===============
# Load some cmake modules.
# 1) define what flags should be available,
# 2) package by name finder
# 3) testing methods
# 4) build configurator (include path, compiler flags, etc.
# ===============
include (flags)
include (definitions)
include (testing)
include (bindings)

# ===============
# library header files
# ===============
include_directories ( "${CMAKE_CURRENT_SOURCE_DIR}/include/" )

# ===============
# compile gcl:
# ===============
add_library( gcl src/GCL.cpp include/gridtools/communication/GCL.hpp include/gridtools/communication/high-level/stats_collector.hpp )
set_target_properties(gcl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")

set(exe_LIBS ${exe_LIBS} gcl)

# ===============
# compile c_bindings_generator:
# ===============

add_library(c_bindings_handle src/c_bindings/handle.cpp)
set_target_properties(c_bindings_handle PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")

add_library(regression_fixture src/tools/regression_fixture.cpp)
set_target_properties(regression_fixture PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")

set(exe_LIBS ${exe_LIBS} c_bindings_handle)

set(CMAKE_FORTRAN_MODOUT_FLAG ON)
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/mod)

add_library(c_bindings_handle_fortran src/c_bindings/handle.f90)
target_link_libraries(c_bindings_handle_fortran c_bindings_handle)
target_include_directories(c_bindings_handle_fortran PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

add_library(c_bindings_generator src/c_bindings/generator.cpp)
set_target_properties(c_bindings_generator PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")

set(exe_LIBS ${exe_LIBS} c_bindings_generator)

add_library(c_bindings_generator_main src/c_bindings/generator_main.cpp)
target_link_libraries(c_bindings_generator_main c_bindings_generator)

add_library(array_descriptor src/c_bindings/array_descriptor.f90)

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

# Add all targets to the build-tree export set
export(TARGETS gcl
  FILE "${PROJECT_BINARY_DIR}/cmake/GridToolsTargets.cmake")

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

# TODO this needs to be cleaned up when cleaning up the CMakeLists (FindGridtools)
# We need to define which files go where (e.g. bindings.cmake)
install(DIRECTORY "include" DESTINATION "${CMAKE_INSTALL_PREFIX}" )
install(TARGETS
        gcl
        gtest
        c_bindings_generator
        c_bindings_handle
        c_bindings_handle_fortran
        c_bindings_generator_main
        array_descriptor
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" EXPORT GridTools )
# We need to install the whole directory because filenames may be upper or lower case (cray compiler).
# It exists a flag to generate lower-case file names with cray, but it will segfault when using them.
install(DIRECTORY
    "${PROJECT_BINARY_DIR}/mod/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gridtools")
install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/bindings.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}")

if ( INSTALL_GT_EXAMPLES )
    install(DIRECTORY
            gt_examples
            DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/src" )
endif()

# Install the export set for use with the install-tree
# TODO should this really be installed to PROJECT_BINARY_DIR (why not install?)
install(EXPORT GridTools DESTINATION
    "${PROJECT_BINARY_DIR}/cmake" COMPONENT dev)
message("compile definitions: ${COMPILE_DEFINITIONS}")

# ===========
# Apply final actions
# ===========
include (post_config_actions)

endif()
