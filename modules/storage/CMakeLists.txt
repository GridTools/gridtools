cmake_policy(SET CMP0048 NEW)

project(gridtools_storage)

cmake_minimum_required(VERSION 3.0)
set (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
set (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

if(POLICY CMP0060)
  # Policy to avoid cmake to substitute libraries with paths and extensions with -l<libname>
  cmake_policy(SET CMP0060 NEW)
endif()

# =======
# Options
# =======

include(version)
include(gridtools_add_option)

## Options and configuration ##
gridtools_add_option( ENABLE_CUDA    DEFAULT ON  DESCRIPTION "Compile with GPU support (CUDA)" )
gridtools_add_option( ENABLE_WERROR  DEFAULT OFF DESCRIPTION "Treat warnings as errors" )
set( GPU_DEVICE "GPU_DEVICE-NOTFOUND" CACHE STRING "GPU device (e.g. K40, K80, P100)" )

# Option to suppress pragma messages (to be reviewed to implement differently) ##
set( GRIDTOOLS_SUPPRESS_MESSAGES "ON" CACHE BOOL "suppress compilation messages")
if(GRIDTOOLS_SUPPRESS_MESSAGES)
  list( APPEND GRIDTOOLS_DEFINITIONS "-DSUPPRESS_MESSAGES" )
endif()

# ============
# Dependencies
# ============

## Boost ##
include(Boost)

## CUDA ##
if( ENABLE_CUDA )
  include(CUDA)
endif()

## C++11 ##
if(${CMAKE_VERSION} VERSION_LESS "3.9.0") 
  if( CMAKE_CXX_COMPILER_ID MATCHES Cray )
    set( CXX11_flag "-hstd=c++11" )
  else()
    set( CXX11_flag "-std=c++11" )
  endif()
  message( STATUS "C++11 flag: ${CXX11_flag}  ( Consider CMake v3.9.0 to autodetect C++11 flag )" )
  list( APPEND GRIDTOOLS_CXX_FLAGS ${CXX11_flag} )
else()
  list( APPEND GRIDTOOLS_COMPILE_FEATURES "cxx_std_11" )
endif()

# =======
# Targets
# =======

## gridtools_common ##
add_library( gridtools_common INTERFACE )
target_include_directories( gridtools_common INTERFACE
  ${GRIDTOOLS_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include/gridtools>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/gridtools> )
target_include_directories( gridtools_common SYSTEM INTERFACE
  ${GRIDTOOLS_SYSTEM_INCLUDE_DIRS} )
target_compile_definitions( gridtools_common INTERFACE ${GRIDTOOLS_DEFINITIONS} )
target_compile_features( gridtools_common INTERFACE ${GRIDTOOLS_COMPILE_FEATURES} )
target_compile_options( gridtools_common INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${GRIDTOOLS_CXX_FLAGS}> )
target_link_libraries( gridtools_common INTERFACE ${GRIDTOOLS_LINK_LIBRARIES} )


## gridtools_storage ##
add_library( gridtools_storage INTERFACE )
target_link_libraries( gridtools_storage INTERFACE gridtools_common )

# =======
# Summary
# =======

if( GRIDTOOLS_DEFINITIONS )
  message( STATUS "Exported definitions: ")
  foreach( _def ${GRIDTOOLS_DEFINITIONS} )
    message( STATUS "  ${_def}")
  endforeach()
endif()

message( STATUS "Exported include directories: ")
message( STATUS "  \${GRIDTOOLS_STORAGE_PATH}/include" )
message( STATUS "  \${GRIDTOOLS_STORAGE_PATH}/include/gridtools (deprecated)" )
foreach( _dir ${GRIDTOOLS_INCLUDE_DIRS} )
  message( STATUS "  ${_dir}")
endforeach()
foreach( _dir ${GRIDTOOLS_SYSTEM_INCLUDE_DIRS} )
  message( STATUS "  (system) ${_dir}")
endforeach()

message( STATUS "Exported compile features: " )
message( STATUS "  cxx_std_11")
foreach( _flag ${GRIDTOOLS_CXX_FLAGS} )
  message( STATUS "  ${_flag}")
endforeach()

if( ENABLE_CUDA AND NOT GRIDTOOLS_HAVE_CUDA )
  message( STATUS "!!! GridTools installation does not support CUDA !!!" )
elseif( GRIDTOOLS_HAVE_CUDA )
  message( STATUS "GridTools installation supports CUDA" )
endif()

# ======
# Export
#=======

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export( PACKAGE gridtools_storage )
export( TARGETS gridtools_common gridtools_storage FILE cmake/gridtools_storage-exports.cmake )

install( TARGETS gridtools_common gridtools_storage EXPORT gridtools_storage-exports DESTINATION share/gridtools/cmake/ )
install( EXPORT gridtools_storage-exports DESTINATION share/gridtools/cmake )

configure_file(
  gridtools_storage-config.cmake.in
  "${PROJECT_BINARY_DIR}/cmake/gridtools_storage-config.cmake"
  @ONLY )


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/gridtools_storage-config-version.cmake"
  VERSION ${GRIDTOOLS_VERSION}
  COMPATIBILITY AnyNewerVersion
)
  
# Install the gridtools_storage-config.cmake and gridtools_storage-config-version.cmake"
install(
  FILES
    "${PROJECT_BINARY_DIR}/cmake/gridtools_storage-config.cmake"
    "${PROJECT_BINARY_DIR}/cmake/gridtools_storage-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gridtools/cmake" )

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/gridtools/common"  DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gridtools" )
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/gridtools/storage" DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gridtools" )
