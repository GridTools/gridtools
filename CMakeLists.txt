cmake_policy(SET CMP0048 NEW)
project(GridTools VERSION "0.1")
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

# ===============
# library header files
# ===============
include_directories ( "${CMAKE_CURRENT_SOURCE_DIR}/include" )

# ===============
# compile gcl:
# ===============
add_library( gcl src/GCL.cpp include/communication/GCL.hpp include/communication/high-level/stats_collector.hpp )
set_target_properties(gcl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")

set(exe_LIBS ${exe_LIBS} gcl)

# ===============
# compile c_bindings_generator:
# ===============
add_library(c_bindings_generator src/c_bindings/generator.cpp src/c_bindings/handle.cpp)
set_target_properties(c_bindings_generator PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")

set(exe_LIBS ${exe_LIBS} c_bindings_generator)

# ===============
# add all cpp files in src:
# ===============
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM sources "${CMAKE_CURRENT_SOURCE_DIR}/src/GCL.cpp")
foreach( src_file ${sources} )
    get_filename_component (exec ${src_file} NAME_WE )
    add_executable (${exec} ${src_file})
    target_link_libraries(${exec} ${exe_LIBS})
endforeach(src_file)

# ===============
# examples and tests
# ===============
include (fix_test_case_name)
if(NOT DISABLE_TESTING)
  add_subdirectory(examples)
  add_subdirectory(unit_tests)
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

install(DIRECTORY "include" DESTINATION "${CMAKE_INSTALL_PREFIX}" )
install(TARGETS "gcl" DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" EXPORT GridTools )
# Install the export set for use with the install-tree
install(EXPORT GridTools DESTINATION
    "${PROJECT_BINARY_DIR}/cmake" COMPONENT dev)
message("compile definitions: ${COMPILE_DEFINITIONS}")

# ===========
# Apply final actions
# ===========
include (post_config_actions)
