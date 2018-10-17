## set suppress messages ##
if(SUPPRESS_MESSAGES)
    add_definitions(-DSUPPRESS_MESSAGES)
endif(SUPPRESS_MESSAGES)

## set verbose mode ##
if(VERBOSE)
    add_definitions(-DVERBOSE)
endif(VERBOSE)

## enable boost variadic PP
## (for nvcc this is not done automatically by boost as it is no tested compiler)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_PP_VARIADICS=1")

## set boost fusion sizes ##
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFUSION_MAX_VECTOR_SIZE=${BOOST_FUSION_MAX_SIZE}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFUSION_MAX_MAP_SIZE=${BOOST_FUSION_MAX_SIZE}")

## enable -Werror
if( WERROR )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Werror" )
endif()

## structured grids ##
if(STRUCTURED_GRIDS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DSTRUCTURED_GRIDS" )
endif()

find_package( Boost 1.58 REQUIRED )

if(Boost_FOUND)
  # HACK: manually add the includes with -isystem because CMake won't respect the SYSTEM flag for CUDA
  foreach(dir ${Boost_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem${dir}")
  endforeach()
  set(exe_LIBS "${Boost_LIBRARIES}" "${exe_LIBS}")
endif()

if(NOT ENABLE_CUDA AND NOT ENABLE_MIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")
endif()

## clang tools ##
find_package(ClangTools)

## gnu coverage flag ##
if(GNU_COVERAGE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
message (STATUS "Building executables for coverage tests")
endif()

## gnu profiling information ##
if(GNU_PROFILE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs")
message (STATUS "Building profiled executables")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=${CXX_STANDARD}")

if(ENABLE_HOST)
  set(HOST_BACKEND_DEFINE "BACKEND_HOST")
endif(ENABLE_HOST)

## cuda support ##
if( ENABLE_CUDA )
  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DGT_CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR}")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DGT_CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}")
  string(REPLACE "." "" CUDA_VERSION ${CUDA_VERSION})
  if( ${CUDA_VERSION} VERSION_LESS "80" )
    message(ERROR " CUDA 7.X or lower is not supported")
  endif()
  if( WERROR )
     #unfortunately we cannot treat all errors as warnings, we have to specify each warning; the only supported warning in CUDA8 is cross-execution-space-call
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --Werror cross-execution-space-call -Xptxas --warning-as-error --nvlink-options --warning-as-error" )
  endif()
  set(CUDA_PROPAGATE_HOST_FLAGS ON)
  set(GPU_SPECIFIC_FLAGS "-D_USE_GPU_ -D_GCL_GPU_")
  set( CUDA_ARCH "sm_35" CACHE STRING "Compute capability for CUDA" )

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
  set(exe_LIBS  ${exe_LIBS} ${CUDA_CUDART_LIBRARY} )
  set (CUDA_LIBRARIES "")
  # adding the additional nvcc flags
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-arch=${CUDA_ARCH}")

  # suppress because of a warning coming from gtest.h
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcudafe" "--diag_suppress=code_is_unreachable")

  if( ${CUDA_VERSION_MAJOR} GREATER_EQUAL 9 )
    # suppress because of boost::fusion::vector ctor
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcudafe" "--diag_suppress=esa_on_defaulted_function_ignored")
  endif()

  if ("${CUDA_HOST_COMPILER}" MATCHES "(C|c?)lang")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_CLANG_SPECIFIC_OPTIONS}")
  endif()

  # workaround for boost::optional with CUDA9.2
  if( (${CUDA_VERSION_MAJOR} EQUAL 9 AND ${CUDA_VERSION_MINOR} EQUAL 2) OR (${CUDA_VERSION_MAJOR} EQUAL 10) )
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-DBOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-DBOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE")
  endif()
  
  if(${CXX_STANDARD} STREQUAL "c++14")
    # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--expt-relaxed-constexpr")
  elseif(${CXX_STANDARD} STREQUAL "c++17")
    message(FATAL_ERROR "c++17 is not supported for CUDA compilation")
  endif()

  set(CUDA_BACKEND_DEFINE "BACKEND_CUDA")
else()
  set (CUDA_LIBRARIES "")
endif()

if( ENABLE_MIC )
    set(MIC_BACKEND_DEFINE "BACKEND_MIC")
endif( ENABLE_MIC )

## clang ##
if((CUDA_HOST_COMPILER MATCHES "(C|c?)lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "(C|c?)lang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth-1024")
    # disable failed vectorization warnings for OpenMP SIMD loops
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pass-failed")
endif()

## Intel compiler ##
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # fix buggy Boost MPL config for Intel compiler (last confirmed with Boost 1.65 and ICC 17)
    # otherwise we run into this issue: https://software.intel.com/en-us/forums/intel-c-compiler/topic/516083
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_MPL_AUX_CONFIG_GCC_HPP_INCLUDED -DBOOST_MPL_CFG_GCC='((__GNUC__ << 8) | __GNUC_MINOR__)'")
    # force boost to use decltype() for boost::result_of, required to compile without errors (ICC 17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_RESULT_OF_USE_DECLTYPE")
    # slightly improve performance
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-subscript-in-range -qoverride-limits")
    # disable failed vectorization warnings for OpenMP SIMD loops
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=15518,15552")
endif()


if(CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
    # Controls preprocessor expansion of macros in Fortran source code.
    set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -eF")
endif()

find_package( OpenMP )

## openmp ##
if(OPENMP_FOUND)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
    set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}" )
endif()

## performance meters ##
if(ENABLE_PERFORMANCE_METERS)
    add_definitions(-DENABLE_METERS)
endif(ENABLE_PERFORMANCE_METERS)

# always use lpthread as cc/ld flags
# be careful! deleting this flags impacts performance
# (even on single core and without pragmas).
set ( exe_LIBS -lpthread ${exe_LIBS} )

## precision ##
if(SINGLE_PRECISION)
  if(ENABLE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DFLOAT_PRECISION=4")
  endif()
  add_definitions("-DFLOAT_PRECISION=4")
  message(STATUS "Computations in single precision")
else()
  if(ENABLE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DFLOAT_PRECISION=8")
  endif()
  add_definitions("-DFLOAT_PRECISION=8")
  message(STATUS "Computations in double precision")
endif()

## mpi ##
if( USE_MPI )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GCL_MPI_")
  find_package(MPI)
  # only test for C++, Fortran MPI is not required
  if (NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "Could not find MPI")
  endif()
  if ( MPI_CXX_INCLUDE_PATH )
      include_directories( "${MPI_CXX_INCLUDE_PATH}" )
  endif()
  list(APPEND exe_LIBS ${MPI_CXX_LIBRARIES})
endif()

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()

file(WRITE ${TEST_MANIFEST} "# Executed tests with arguments\n")

function(add_to_test_manifest)
    file(APPEND ${TEST_MANIFEST} "${ARGN}\n")
endfunction(add_to_test_manifest)


## test script generator ##
file(WRITE ${TEST_SCRIPT} "#!/bin/sh\n")
file(APPEND ${TEST_SCRIPT} "hostname\n")
file(APPEND ${TEST_SCRIPT} "res=0\n")
function(gridtools_add_test test_name test_script test_exec)
  file(APPEND ${test_script} "echo ${test_exec}" " ${ARGN}" "\n")
  file(APPEND ${test_script} "${test_exec}" " ${ARGN}" "\n")
  file(APPEND ${test_script} "res=$((res || $? ))\n")
  add_to_test_manifest(${test_name} ${ARGN})
endfunction(gridtools_add_test)

## test script generator for MPI tests ##
file(WRITE ${TEST_MPI_SCRIPT} "res=0\n")
function(gridtools_add_mpi_test test_name test_exec)
  file(APPEND ${TEST_MPI_SCRIPT} "echo \$LAUNCH_MPI_TEST ${test_exec}" " ${ARGN}" "\n")
  file(APPEND ${TEST_MPI_SCRIPT} "\$LAUNCH_MPI_TEST ${test_exec}" " ${ARGN}" "\n")
  file(APPEND ${TEST_MPI_SCRIPT} "res=$((res || $? ))\n")
  add_to_test_manifest(${test_name} ${ARGN})
endfunction(gridtools_add_mpi_test)

file(WRITE ${TEST_CUDA_MPI_SCRIPT} "res=0\n")
function(gridtools_add_cuda_mpi_test test_name test_exec)
  file(APPEND ${TEST_CUDA_MPI_SCRIPT} "echo \$LAUNCH_MPI_TEST ${test_exec}" " ${ARGN}" "\n")
  file(APPEND ${TEST_CUDA_MPI_SCRIPT} "\$LAUNCH_MPI_TEST ${test_exec}" " ${ARGN}" "\n")
  file(APPEND ${TEST_CUDA_MPI_SCRIPT} "res=$((res || $? ))\n")
  add_to_test_manifest(${test_name} ${ARGN})
endfunction(gridtools_add_cuda_mpi_test)

## caching ##
if( NOT ENABLE_CACHING )
    add_definitions( -D__DISABLE_CACHING__ )
endif()

add_definitions(-DGTEST_COLOR )
include_directories( ${GTEST_INCLUDE_DIR} )
include_directories( ${GMOCK_INCLUDE_DIR} )
