cmake_minimum_required( VERSION 2.8 )
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project( GRIDTOOLS )
enable_language( CXX )

enable_testing()

# Not sure why it's desirable to specify the build directory, but it seems
# to break testing with MSVC, so disabling it conditionally.
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(EXECUTABLE_OUTPUT_PATH "../build")
endif()

include_directories( include/ ) ## Library header files

# Find boost
if(WIN32)
  # Auto-linking happens on Windows, so we don't need to specify specific components
  find_package( Boost REQUIRED )
else()
  # On other platforms, me must be specific about which libs are required
  find_package( Boost COMPONENTS timer system chrono REQUIRED )
endif()

# If the version is less than 1.56, we need a special version of Fusion
if (Boost_VERSION LESS 105600)
  set( GPU_ENABLED_FUSION "../fusion/include" CACHE STRING "Path to...")
  include_directories( SYSTEM ${GPU_ENABLED_FUSION} ) ## Library header files
endif()

# include boost
include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

# Optionally include CUDA
set( USE_GPU "OFF" CACHE BOOL "Compile with GPU support (CUDA)" )
if( "${USE_GPU}" STREQUAL "ON" )
  find_package(CUDA REQUIRED)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_GPU_")
else()
  set (CUDA_LIBRARIES "")
endif()

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

# find GTest
find_package(GTest)
if (GTEST_FOUND)
  include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
else()
  message("GTest not found. GTest targets will not be compiled. (Try setting GTEST_ROOT in the env.)")
endif()

# Set some project-wide MSVC compile flags
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  add_definitions( -D_VARIADIC_MAX=10 )
endif()

if(OPENMP_FOUND)
    set (PAPI_WRAP_MODULE "OFF" CACHE BOOL "If on, the papi-wrap library is compiled with the project")
    if ( "${PAPI_WRAP_MODULE}" STREQUAL "ON" )
        add_subdirectory( include/external/perfcount )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PAPI_WRAP")
    endif()
endif()

add_subdirectory( examples )
add_subdirectory( unit_tests )

set( PAPI_WRAP_AS_MODULE "ON" )

