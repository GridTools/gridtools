cmake_minimum_required( VERSION 2.8 )
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project( GRIDTOOLS )
enable_language( CXX )

enable_testing()

set(EXECUTABLE_OUTPUT_PATH "../build")

include_directories( include/ ) ## Library header files
include_directories( ${GPU_ENABLED_FUSION} ) ## Library header files

# include boost
set( GPU_ENABLED_FUSION "../fusion/include" CACHE STRING "Path to...")
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

# Optionally include CUDA
set( USE_GPU "OFF" CACHE BOOL "Compile with GPU support (CUDA)" )
if( "${USE_GPU}" STREQUAL "ON" )
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_GPU_")
else()
  set (CUDA_LIBRARIES "")
endif()

# find GTest
find_package(GTest)
if (GTEST_FOUND)
   include_directories(${GTEST_INCLUDE_DIRS})
else()
   message("GTest not found. GTest targets will not be compiled. (Try setting GTEST_ROOT in the env.)")
endif()

# Set some project-wide MSVC compile flags
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_VARIADIC_MAX=10")
endif()

add_subdirectory( examples )
add_subdirectory( unit_tests )
