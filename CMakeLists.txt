cmake_minimum_required( VERSION 2.8 )
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project( GRIDTOOLS )
enable_language( CXX )


## enable to use ctest:
enable_testing()
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")
message("cmake module path " ${CMAKE_MODULE_PATH})

if(GNU_COVERAGE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lgcov" )
message ("building executables for coverage tests")
endif()

if(GNU_PROFILE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs")
message ("building profiled executables")
endif()


# Not sure why it's desirable to specify the build directory, but it seems
# to break testing with MSVC, so disabling it conditionally.
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(EXECUTABLE_OUTPUT_PATH "../build")
endif()

include_directories( include/ include/communication/high-level ) ## Library header files

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
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -Xcudafe \"--diag_suppress=dupl_calling_convention\" -Xcudafe \"--diag_suppress=code_is_unreachable\" -Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\" -Xcudafe \"--diag_suppress=calling_convention_not_allowed\" -Xcudafe \"--diag_suppress=conflicting_calling_conventions\"")
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

set ( exe_LIBS ${Boost_LIBRARIES} )

if(OPENMP_FOUND)
    set (PAPI_WRAP_LIBRARY "OFF" CACHE BOOL "If on, the papi-wrap library is compiled with the project")
endif()

################# look for papi_wrap #############################
if ( PAPI_WRAP_LIBRARY )
  find_package(PapiWrap)
  if ( PAPI_WRAP_FOUND )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PAPI_WRAP" )
    set( PAPI_WRAP_MODULE "ON" )
    include_directories( "${PAPI_WRAP_INCLUDE_DIRS}" )
    set ( exe_LIBS "${exe_LIBS}" "${PAPI_WRAP_LIBRARIES}" )
  else()
    message ("papi-wrap not found. Please set PAPI_WRAP_PREFIX to the root path of the papi-wrap library. papi-wrap not used!")
  endif()
endif()
##################################################################

################## look for PAPI library ######################
if(USE_PAPI)
  find_package(PAPI REQUIRED)
  if(PAPI_FOUND)
    include_directories( "${PAPI_INCLUDE_DIRS}" )
    set ( exe_LIBS "${exe_LIBS}" "${PAPI_LIBRARIES}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PAPI" )
  else()
    message("PAPI library not found. set the PAPI_PREFIX")
  endif()
endif()
##################################################################

set( GCL_MPI "${USE_MPI}" )
set( GCL_GPU "${USE_GPU}" )
set( GCL_ONLY "OFF" CACHE BOOL "If on only library is build but not the examples and tests" )
set( USE_MPI_COMPILER "OFF" CACHE BOOL "On rosa turn this flag off since compiler takes care of mpi already" )
set( HOST_SPECIFIC_OPTIONS "" CACHE STRING "Options passed only to HOST COMPILER and not ACCELERATOR COMPILER" )


if( "${GCL_GPU}" STREQUAL "ON" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GCL_GPU_")
else()
set (CUDA_LIBRARIES "")
endif()

set( USE_MPI "OFF" CACHE BOOL "Compile with MPI support" )

if( USE_MPI )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GCL_MPI_")
  if( USE_MPI_COMPILER )
    FIND_PACKAGE(MPI REQUIRED)
    INCLUDE(CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER(mpicxx "MPI C++ Compiler")
  endif()
endif()


ADD_LIBRARY( gcl src/GCL.cpp include/communication/GCL.h include/communication/high-level/stats_collector.h)

add_subdirectory( unit_tests )
add_subdirectory( examples )



# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)
