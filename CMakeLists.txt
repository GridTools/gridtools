cmake_minimum_required(VERSION 3.12.4)

cmake_policy(SET CMP0048 NEW)
project(GridTools VERSION 0.0.0 LANGUAGES CXX)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(CTest)

# ===============
# Load some cmake modules.
# 1) define what flags should be available,

include(detect_features)
detect_cuda()
detect_openmp()
detect_mpi()

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
if (BUILD_TESTING)
    include (testing)
endif()
include (bindings)
include (components)

# TODO this install targets are not finished on purpose.
if( COMPONENT_COMMON )
    install(TARGETS gridtools EXPORT GridToolstargets
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      RUNTIME DESTINATION bin
      INCLUDES DESTINATION include
    )
    install(EXPORT GridToolstargets
      FILE GridToolsTargets.cmake
      NAMESPACE gridtools::
      DESTINATION ${INSTALL_CONFIGDIR}
    )

    generate_target_for(NAME common)
    target_link_libraries(common INTERFACE gridtools)
endif()

if( COMPONENT_BOUNDARY_CONDITIONS )
    generate_target_for(NAME boundary-conditions)
    target_link_libraries( boundary-conditions INTERFACE common )
endif()

if( COMPONENT_STENCIL_COMPOSITION )
    generate_target_for(NAME stencil-composition)
    target_link_libraries( stencil-composition INTERFACE common )
endif()

if( COMPONENT_GCL )
    generate_target_for(NAME gcl SOURCES src/GCL.cpp include/gridtools/communication/GCL.hpp include/gridtools/communication/high-level/stats_collector.hpp )
    target_link_libraries(gcl gridtools)
    if( GT_USE_MPI )
        target_link_libraries(gcl MPI::MPI_CXX)
    endif()
endif()

if( COMPONENT_DISTRIBUTED_BOUNDARIES )
    generate_target_for(NAME distributed-boundaries)
    target_link_libraries( distributed-boundaries INTERFACE boundary-conditions gcl )
endif()

if( COMPONENT_STORAGE )
    generate_target_for(NAME storage)
    target_link_libraries( distributed-boundaries INTERFACE common )
endif()

if( COMPONENT_INTERFACE )
    generate_target_for(NAME interface)
endif()

if( COMPONENT_TOOLS )
    generate_target_for(NAME tools)
endif()

if(COMPONENT_C_BINDINGS)
    generate_target_for(NAME c_bindings)
endif()

# ===============
# examples and tests
# ===============
if( GT_INSTALL_EXAMPLES )
  add_subdirectory(gt_examples)
endif()

if(BUILD_TESTING)
  # TODO move to the right directory
  add_library(regression_main src/tools/regression_fixture.cpp)
  target_link_libraries(regression_main gtest gridtools)

  add_subdirectory(regression)
  add_subdirectory(unit_tests)
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

include(export)

# ===========
# Apply final actions
# ===========
include (post_config_actions)
