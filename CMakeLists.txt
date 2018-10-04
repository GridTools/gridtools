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

### The followinf function appends "install-" string to the list of arguments following 
### the name of the variable you want to get the outout in.
function(prepend_install outvar)
   set(outlist "")
   foreach(elem ${ARGN})
      list(APPEND outlist "install-${elem}")
   endforeach(elem)
   set(${outvar} ${outlist} PARENT_SCOPE)
endfunction(prepend_install)

### The following function works like this:
###
### First argument the name of the library to build
###
### All other arguments are the sources for compiling the library. If this list is empty this will be a header-only library!
###
function(generate_target_for name)
#    message(${ARGN} ${name}
    if ("${ARGN} " STREQUAL " ")
        add_library(${name} INTERFACE)
        target_include_directories(${name} INTERFACE $<BUILD_INTERFACE:"${CMAKE_CURRENT_SOURCE_DIR}/include/"> $<INSTALL_INTERFACE:"${CMAKE_CURRENT_SOURCE_DIR}/include/"> )

    else ()
        add_library(${name} ${ARGN})
        target_include_directories(${name}
                     PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                        $<INSTALL_INTERFACE:include>
                     PRIVATE
                        ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
    endif()
endfunction(generate_target_for)

### The following function works like this:
###
### First argument the name of the library as exported
###
### Second is the folder in `include/gridtools` where the headers to be installed are
###
### The Third argument the library upon which it depends (usually "common")
###
function(generate_install_targets_for name folder depend)
    install(TARGETS ${name} EXPORT ${name}targets
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      RUNTIME DESTINATION bin
      INCLUDES DESTINATION include
      COMPONENT ${name}component
    )
    install(EXPORT ${name}targets
      FILE ${name}Targets.cmake
      NAMESPACE ${name}::
      DESTINATION lib/cmake/${name}
    )

    install(DIRECTORY "include/gridtools/${folder}" DESTINATION include COMPONENT ${name}component )
    if ("${depend} " STREQUAL "None ")
        add_custom_target(install-${name}
                          COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=${name}component -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    else()
        string(STRIP ${depend} depend)
        string(REPLACE " " ";" depend ${depend})
        prepend_install(depend ${depend})
        string(REPLACE ";" " " depend "${depend}")
        separate_arguments(depend)
        add_custom_target(install-${name}
                          DEPENDS ${depend} ${name}
                          COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=${name}component -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    endif()

endfunction(generate_install_targets_for)

generate_target_for(common)
generate_install_targets_for(common common None)

generate_target_for(boundary-conditions)
generate_install_targets_for(boundary-conditions boundary-conditions common)

generate_target_for(distributed-boundaries)
generate_install_targets_for(distributed-boundaries distributed-boundaries "common boundary-conditions")

generate_target_for(stencil-composition)
generate_install_targets_for(stencil-composition stencil-composition common)


# ===============
# compile gcl:
# ===============
generate_target_for(gcl src/GCL.cpp include/gridtools/communication/GCL.hpp include/gridtools/communication/high-level/stats_collector.hpp )
generate_install_targets_for(gcl communication common)

set(exe_LIBS ${exe_LIBS} gcl)

# ===============
# compile c_bindings_generator:
# ===============

add_library(c_bindings_handle src/c_bindings/handle.cpp)
set_target_properties(c_bindings_handle PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")
target_include_directories(c_bindings_handle
                     PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                        $<INSTALL_INTERFACE:include>
                     PRIVATE
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/c_bindings
)

set(exe_LIBS ${exe_LIBS} c_bindings_handle)

set(CMAKE_FORTRAN_MODOUT_FLAG ON)
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/mod)

add_library(c_bindings_handle_fortran src/c_bindings/handle.f90)
target_link_libraries(c_bindings_handle_fortran c_bindings_handle)
target_include_directories(c_bindings_handle_fortran PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

add_library(c_bindings_generator src/c_bindings/generator.cpp)
set_target_properties(c_bindings_generator PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS}")
target_include_directories(c_bindings_generator
                     PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                        $<INSTALL_INTERFACE:include>
                     PRIVATE
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/c_bindings
)

set(exe_LIBS ${exe_LIBS} c_bindings_generator)

add_library(c_bindings_generator_main src/c_bindings/generator_main.cpp)
target_link_libraries(c_bindings_generator_main c_bindings_generator)

add_library(array_descriptor src/c_bindings/array_descriptor.f90)

# ===============
# examples and tests
# ===============
include (fix_test_case_name)
if(NOT DISABLE_TESTING)
  add_subdirectory(regression)
  add_subdirectory(unit_tests)
endif()

if( INSTALL_GT_EXAMPLES )
   SET(COMPILE_EXAMPLES ON)
endif()

if( COMPILE_EXAMPLES )
  add_subdirectory(gt_examples)
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


####REMOVE install(TARGETS gcl EXPORT libgcl COMPONENT GCL DESTINATION lib INCLUDES DESTINATION include)
####REMOVE add_custom_target(install-gcl DEPENDS gcl install-common COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=GCL -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
####REMOVE install(DIRECTORY "include/gridtools/communication" DESTINATION include COMPONENT GCL )
####REMOVE generate_export_header(gcl)

####REMOVE include(CMakePackageConfigHelpers)
####REMOVE write_basic_package_version_file(
####REMOVE   "${CMAKE_CURRENT_BINARY_DIR}/hello/LibGCLConfigVersion.cmake"
####REMOVE   VERSION ${Upstream_VERSION}
####REMOVE   COMPATIBILITY AnyNewerVersion
####REMOVE )

####REMOVE export(EXPORT libgcl
####REMOVE   FILE "${CMAKE_CURRENT_BINARY_DIR}/hello/LibGCLTargets.cmake"
####REMOVE   NAMESPACE Upstream::
####REMOVE )

####REMOVE configure_file(GridToolsConfig.cmake.in
####REMOVE   "${PROJECT_BINARY_DIR}/cmake/GridToolsConfig.cmake" @ONLY)
####REMOVE configure_file(GridToolsConfigVersion.cmake.in
####REMOVE   "${PROJECT_BINARY_DIR}/cmake/GridToolsConfigVersion.cmake" @ONLY)

####REMOVE # Install the GridToolsConfig.cmake and GridToolsConfigVersion.cmake
####REMOVE install(FILES "${PROJECT_BINARY_DIR}/cmake/GridToolsConfig.cmake"
####REMOVE   "${PROJECT_BINARY_DIR}/cmake/GridToolsConfigVersion.cmake"
####REMOVE   DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT dev)

####REMOVE # TODO this needs to be cleaned up when cleaning up the CMakeLists (FindGridtools)
####REMOVE # We need to define which files go where (e.g. bindings.cmake)
####REMOVE install(DIRECTORY "include" DESTINATION "${CMAKE_INSTALL_PREFIX}" )
####REMOVE install(TARGETS
####REMOVE         gcl
####REMOVE         gtest
####REMOVE         c_bindings_generator
####REMOVE         c_bindings_handle
####REMOVE         c_bindings_handle_fortran
####REMOVE         c_bindings_generator_main
####REMOVE         array_descriptor
####REMOVE     DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" EXPORT GridTools )
####REMOVE install(FILES
####REMOVE     "${PROJECT_BINARY_DIR}/mod/gt_handle.mod"
####REMOVE     "${PROJECT_BINARY_DIR}/mod/array_descriptor.mod"
####REMOVE     DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gridtools")
####REMOVE install(FILES
####REMOVE     "${PROJECT_SOURCE_DIR}/cmake/bindings.cmake"
####REMOVE     DESTINATION "${CMAKE_INSTALL_PREFIX}")

####REMOVE if ( INSTALL_GT_EXAMPLES )
####REMOVE     install(DIRECTORY
####REMOVE             gt_examples
####REMOVE             DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/src" )
####REMOVE endif()

####REMOVE # Install the export set for use with the install-tree
####REMOVE # TODO should this really be installed to PROJECT_BINARY_DIR (why not install?)
####REMOVE install(EXPORT GridTools DESTINATION
####REMOVE     "${PROJECT_BINARY_DIR}/cmake" COMPONENT dev)
####REMOVE message("compile definitions: ${COMPILE_DEFINITIONS}")

# ===========
# Apply final actions
# ===========
include (post_config_actions)

endif()
