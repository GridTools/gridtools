cmake_policy(SET CMP0048 NEW)

project(gridtools_storage)

cmake_minimum_required(VERSION 3.3)
# CMake Version 3.3 adds generator_expression $<COMPILE_LANGUAGE:CXX>

include( ../cmake/gridtools.cmake )

# =============
# Options
# =============

## Options and configuration ##
gridtools_add_option( ENABLE_CUDA    DEFAULT ON  DESCRIPTION "Compile with GPU support (CUDA)" )
gridtools_add_option( ENABLE_WERROR  DEFAULT OFF DESCRIPTION "Treat warnings as errors" )
set( GPU_DEVICE "GPU_DEVICE-NOTFOUND" CACHE STRING "GPU device (e.g. K40, K80, P100)" )

# Option to suppress pragma messages (to be reviewed to implement differently) ##
set( GRIDTOOLS_SUPPRESS_MESSAGES "ON" CACHE BOOL "suppress compilation messages")
if(GRIDTOOLS_SUPPRESS_MESSAGES)
  list( APPEND GRIDTOOLS_DEFINITIONS "-DSUPPRESS_MESSAGES" )
endif()

# =============
# Dependencies
# =============

## Boost ##
gridtools_setup_Boost()

## CUDA ##
if( ENABLE_CUDA )
  gridtools_setup_CUDA()
endif()

## C++11 ##
gridtools_setup_cxx()

# =============
# Targets
# =============

## gridtools_common ##
add_library( gridtools_common INTERFACE )
target_include_directories( gridtools_common INTERFACE
  $<BUILD_INTERFACE:${GRIDTOOLS_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${GRIDTOOLS_SOURCE_DIR}/include/gridtools>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/gridtools> )
target_include_directories( gridtools_common SYSTEM INTERFACE ${GRIDTOOLS_SYSTEM_INCLUDE_DIRS} )
target_compile_definitions( gridtools_common INTERFACE ${GRIDTOOLS_DEFINITIONS} )
target_compile_features( gridtools_common INTERFACE ${GRIDTOOLS_COMPILE_FEATURES} )
target_compile_options( gridtools_common INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${GRIDTOOLS_CXX_FLAGS}> )
target_link_libraries( gridtools_common INTERFACE ${GRIDTOOLS_LINK_LIBRARIES} )


## gridtools_storage ##
add_library( gridtools_storage INTERFACE )
target_link_libraries( gridtools_storage INTERFACE gridtools_common )

# =============
# Summary
# =============

if( GRIDTOOLS_DEFINITIONS )
  message( STATUS "Exported definitions: ")
  foreach( _def ${GRIDTOOLS_DEFINITIONS} )
    message( STATUS "  ${_def}")
  endforeach()
endif()

message( STATUS "Exported include directories: ")
message( STATUS "  \${GRIDTOOLS_SOURCE_DIR}/include" )
message( STATUS "  \${GRIDTOOLS_SOURCE_DIR}/include/gridtools (deprecated)" )
foreach( _dir ${GRIDTOOLS_SYSTEM_INCLUDE_DIRS} )
  message( STATUS "  (system) ${_dir}")
endforeach()

message( STATUS "Exported C++ compile features: " )
message( STATUS "  cxx_std_11")
foreach( _flag ${GRIDTOOLS_CXX_FLAGS} )
  message( STATUS "  ${_flag}")
endforeach()

if( ENABLE_CUDA AND NOT GRIDTOOLS_HAVE_CUDA )
  message( STATUS "!!! GridTools installation does not support CUDA !!!" )
elseif( GRIDTOOLS_HAVE_CUDA )
  message( STATUS "GridTools installation supports CUDA with CUDA_ARCH=${CUDA_ARCH}" )
  message( STATUS "Exported nvcc compile flags: " )
  foreach( _flag ${GRIDTOOLS_NVCC_FLAGS} )
    message( STATUS "  ${_flag}")
  endforeach()
endif()

# =============
# Export
# =============

gridtools_cmake_workarounds()
gridtools_export( gridtools_common gridtools_storage )

# =============
# Tests
# =============

# Todo