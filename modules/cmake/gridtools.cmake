# gridtools.cmake
#
# Sets variables
#  - GRIDTOOLS_VERSION
#  - GRIDTOOLS_SOURCE_DIR
#  - UPPERCASE_PROJECT_NAME ( uppercase version of PROJECT_NAME )

set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_LIST_DIR}" )

include( gridtools_policies )
include( gridtools_add_option )
include( gridtools_add_nvcc_flags )
include( gridtools_setup_Boost )
include( gridtools_setup_CUDA )
include( gridtools_setup_cxx )
include( gridtools_export )
include( gridtools_cmake_workarounds )

set( GRIDTOOLS_VERSION 0.1 )

STRING( TOUPPER ${PROJECT_NAME} UPPERCASE_PROJECT_NAME )
get_filename_component( GRIDTOOLS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE )

message( STATUS "---------------------------------------------------------" )
message( STATUS "[${PROJECT_NAME}] (${GRIDTOOLS_VERSION})" )
