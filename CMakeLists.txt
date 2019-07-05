cmake_minimum_required(VERSION 3.14.5)

cmake_policy(SET CMP0048 NEW)
file(STRINGS "version.txt" __GT_VERSION)
project(GridTools VERSION ${__GT_VERSION} LANGUAGES CXX)
unset(__GT_VERSION)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(CTest)

# ===============
# Load some cmake modules.
# 1) define what flags should be available,

include(detect_features)
detect_cuda()
detect_openmp()
detect_mpi()

include(flags)

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
include(definitions)
if (BUILD_TESTING)
    include(testing)
endif()
include(components)

include(FetchContent)
FetchContent_Declare(
  cpp_bindgen
  GIT_REPOSITORY https://github.com/havogt/cpp_bindgen.git
  GIT_TAG        add_install_function
)
# Enable legacy mode for CPP_BINDGEN for users of GT 1.0
# TODO remove in GT 2.0
set(CPP_BINDGEN_GT_LEGACY ON)
FetchContent_MakeAvailable(cpp_bindgen)

# TODO this install targets are not finished on purpose.
if( COMPONENT_COMMON )
    generate_target_for(NAME common)
    target_link_libraries(common INTERFACE gridtools)
endif()

if( COMPONENT_BOUNDARY_CONDITIONS )
    generate_target_for(NAME boundary_conditions)
    target_link_libraries(boundary_conditions INTERFACE common)
endif()

if( COMPONENT_STENCIL_COMPOSITION )
    generate_target_for(NAME stencil_composition)
    target_link_libraries(stencil_composition INTERFACE common)
endif()

if( COMPONENT_GCL )
    generate_target_for(NAME gcl SOURCES src/GCL.cpp include/gridtools/communication/GCL.hpp include/gridtools/communication/high_level/stats_collector.hpp )
    target_link_libraries(gcl gridtools)
    if( GT_USE_MPI )
        target_link_libraries(gcl MPI::MPI_CXX)
    endif()
endif()

if( COMPONENT_DISTRIBUTED_BOUNDARIES )
    generate_target_for(NAME distributed_boundaries)
    target_link_libraries(distributed_boundaries INTERFACE boundary_conditions gcl)
endif()

if( COMPONENT_STORAGE )
    generate_target_for(NAME storage)
    target_link_libraries(distributed_boundaries INTERFACE common)
endif()

if( COMPONENT_INTERFACE )
    generate_target_for(NAME interface)
endif()

if( COMPONENT_TOOLS )
    generate_target_for(NAME tools)
endif()

# ===============
# tests
# ===============
if(BUILD_TESTING)
  # TODO move to the right directory
  add_library(regression_main src/tools/regression_fixture.cpp)
  target_link_libraries(regression_main gtest gridtools)

  add_subdirectory(regression)
  add_subdirectory(unit_tests)

  add_subdirectory(docs_src/manuals/getting_started)
endif()

# ===============
# examples
# ===============
if(GT_INSTALL_EXAMPLES)
    add_subdirectory(examples)
endif()

# ===============
# Downloads the gridtools_experimental repository
# ===============
if(GT_ENABLE_EXPERIMENTAL_REPOSITORY)
  include(gridtools_experimental/download_gridtools_experimental)
endif()

# ===============
# Python scripts
# ===============
if(GT_ENABLE_PYUTILS)
    add_subdirectory(pyutils)
endif()

# absolute path to avoid collision with "export" from other packages
include(${CMAKE_CURRENT_LIST_DIR}/cmake/export.cmake)
