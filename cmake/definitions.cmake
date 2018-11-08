set(GT_CXX_MANDATORY_FLAGS)    # Flags that are needed to compile GT ap plications correctly
set(GT_CXX_BUILDING_FLAGS)     # Flags needed to compile unit tests and such, but not for export
set(GT_CXX_OPTIONAL_FLAGS)     # Flags that are optional for compiling, like removing warnings and such
set(GT_CXX_OPTIMIZATION_FLAGS) # Flags used for optimization

set(GT_CUDA_MANDATORY_FLAGS)    # Flags that are needed to compile GT ap plications correctly
set(GT_CUDA_BUILDING_FLAGS)     # Flags needed to compile unit tests and such, but not for export
set(GT_CUDA_OPTIONAL_FLAGS)     # Flags that are optional for compiling, like removing warnings and such
set(GT_CUDA_OPTIMIZATION_FLAGS) # Flags used for optimization

set(GT_C_BUILDING_FLAGS)       # Flags for the C components (driver.c)

if("${GT_CXX_STANDARD}" STREQUAL "c++11")
    set (GT_CXX_STANDARD_VALUE 11)
elseif("${GT_CXX_STANDARD}" STREQUAL "c++14")
    set (GT_CXX_STANDARD_VALUE 14)
elseif("${GT_CXX_STANDARD}" STREQUAL "c++17")
    set (GT_CXX_STANDARD_VALUE 17)
else()
    message(FATAL_ERROR "Invalid argument for GT_CXX_STANDARD (`${GT_CXX_STANDARD}`)")
endif()

set(CMAKE_CXX_STANDARD ${GT_CXX_STANDARD_VALUE})
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD ${GT_CXX_STANDARD_VALUE})
set(CMAKE_CUDA_EXTENSIONS OFF)

if(GT_SUPPRESS_MESSAGES)
    set( GT_CXX_BUILDING_FLAGS ${GT_CXX_BUILDING_FLAGS}  -DSUPPRESS_MESSAGES )
    set( GT_C_BUILDING_FLAGS ${GT_C_BUILDING_FLAGS}  -DSUPPRESS_MESSAGES )
endif(GT_SUPPRESS_MESSAGES)

if(GT_VERBOSE)
    set( GT_CXX_BUILDING_FLAGS ${GT_C_BUILDING_FLAGS}  -DVERBOSE )
    set( GT_C_BUILDING_FLAGS ${GT_C_BUILDING_FLAGS}  -DVERBOSE )
endif(GT_VERBOSE)

## enable boost variadic PP
## (for nvcc this is not done automatically by boost as it is no tested compiler)
set( GT_CXX_MANDATORY_FLAGS ${GT_CXX_MANDATORY_FLAGS}  -DBOOST_PP_VARIADICS=1  )

## set boost fusion sizes ##
set( GT_CXX_OPTIONAL_FLAGS ${GT_CXX_OPTIONAL_FLAGS}  -DFUSION_MAX_VECTOR_SIZE=${GT_BOOST_FUSION_MAX_SIZE} )
set( GT_CXX_OPTIONAL_FLAGS ${GT_CXX_OPTIONAL_FLAGS}  -DFUSION_MAX_MAP_SIZE=${GT_BOOST_FUSION_MAX_SIZE} )

if ( (CMAKE_CXX_COMPILER_ID MATCHES "(C|c?)lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel") )
    set(GT_TREAT_WARNINGS_AS_ERROR OFF)
endif()

## enable -Werror
if( GT_TREAT_WARNINGS_AS_ERROR )
    set( CMAKE_CXX_BUILDING_FLAGS ${GT_CXX_BUILDING_FLAGS}  -Werror )
endif()

## structured grids ##
if(STRUCTURED_GRIDS)
    set( GT_CXX_BUILDING_FLAGS ${GT_CXX_BUILDING_FLAGS}  -DSTRUCTURED_GRIDS )
endif()

if(NOT GT_ENABLE_TARGET_CUDA AND NOT GT_ENABLE_TARGET_MC)
    set( GT_CXX_OPTIMIZATION_FLAGS ${GT_CXX_OPTIMIZATION_FLAGS}  -mtune=native -march=native )
endif()

## clang tools ##
find_package(ClangTools)

if(GT_ENABLE_TARGET_X86)
  set(X86_BACKEND_DEFINE "BACKEND_X86")
endif(GT_ENABLE_TARGET_X86)

## cuda support ##
if( GT_ENABLE_TARGET_CUDA )
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR "${CUDA_VERSION}")
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR "${CUDA_VERSION}")

  # this is only needed to get CUDA_CUDART_LIBRARY, please do not use other variables from here!
  # Find a better solution for this (consider https://gitlab.kitware.com/cmake/cmake/issues/17816)
  find_package(CUDA REQUIRED)

  set(GT_CUDA_MANDATORY_FLAGS ${GT_CUDA_MANDATORY_FLAGS} "-DGT_CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR}")
  set(GT_CUDA_MANDATORY_FLAGS ${GT_CUDA_MANDATORY_FLAGS} "-DGT_CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}")
  set(GT_CXX_MANDATORY_FLAGS ${GT_CXX_MANDATORY_FLAGS} "-DGT_CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR}")
  set(GT_CXX_MANDATORY_FLAGS ${GT_CXX_MANDATORY_FLAGS} "-DGT_CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}")
  if( ${CUDA_VERSION} VERSION_LESS "8.0" )
    message(ERROR " CUDA 7.X or lower is not supported")
  endif()
  if( GT_TREAT_WARNINGS_AS_ERROR )
     #unfortunately we cannot treat all as warnings, we have to specify each warning; the only supported warning in CUDA8 is cross-execution-space-call
     # TODO check this...
    set(GT_CUDA_BUILDING_FLAGS ${GT_CUDA_BUILDING_FLAGS} --Werror cross-execution-space-call -Xptxas --warning-as-error --nvlink-options --warning-as-error )
  endif()
  set(CUDA_PROPAGATE_HOST_FLAGS ON)
  set(GPU_SPECIFIC_FLAGS -D_USE_GPU_ -D_GCL_GPU_)
  set( CUDA_ARCH "sm_35" CACHE STRING "Compute capability for CUDA" )

  # adding the additional nvcc flags
  set(GT_CUDA_MANDATORY_FLAGS ${GT_CUDA_MANDATORY_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_ARCH}>)

  # suppress because of a warning coming from gtest.h
  set(GT_CUDA_BUILDING_FLAGS ${GT_CUDA_BUILDING_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=code_is_unreachable>)

  if( ${CUDA_VERSION_MAJOR} GREATER_EQUAL 9 )
    # suppress because of boost::fusion::vector ctor
    set(GT_CUDA_BUILDING_FLAGS ${GT_CUDA_BUILDING_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored>)
  endif()

  if (CMAKE_CXX_COMPILER_ID MATCHES "(C|c?)lang")
      set(GT_CUDA_MANDATORY_FLAGS ${GT_CUDA_MANDATORY_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-ccbin=${CMAKE_CXX_COMPILER}>)
  endif()

  # workaround for boost::optional with CUDA9.2
  if(${CUDA_VERSION} VERSION_GREATER "9.1")
    set(GT_CUDA_MANDATORY_FLAGS ${GT_CUDA_MANDATORY_FLAGS} -DBOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL)
    set(GT_CUDA_MANDATORY_FLAGS ${GT_CUDA_MANDATORY_FLAGS} -DBOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE)
  endif()

  if(${GT_CXX_STANDARD} STREQUAL "c++14")
    # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
    set(GT_CUDA_MANDATORY_FLAGS ${GT_CUDA_MANDATORY_FLAGS} --expt-relaxed-constexpr)
  elseif(${GT_CXX_STANDARD} STREQUAL "c++17")
    message(FATAL_ERROR "c++17 is not supported for CUDA compilation")
  endif()

  set(CUDA_BACKEND_DEFINE "BACKEND_CUDA")
endif()

if( GT_ENABLE_TARGET_MC )
    set(MC_BACKEND_DEFINE "BACKEND_MC")
endif( GT_ENABLE_TARGET_MC )

## clang ##
if((CUDA_HOST_COMPILER MATCHES "(C|c?)lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "(C|c?)lang"))
    set( GT_CXX_HOST_ONLY_FLAGS ${GT_CXX_HOST_ONLY_FLAGS}  -ftemplate-depth-1024 )
    # disable failed vectorization warnings for OpenMP SIMD loops
    set( GT_CXX_HOST_ONLY_FLAGS ${GT_CXX_HOST_ONLY_FLAGS}  -Wno-pass-failed )
endif()

## Intel compiler ##
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # fix buggy Boost MPL config for Intel compiler (last confirmed with Boost 1.65 and ICC 17)
    # otherwise we run into this issue: https://software.intel.com/en-us/forums/intel-c-compiler/topic/516083
    set( GT_CXX_MANDATORY_FLAGS ${GT_CXX_MANDATORY_FLAGS}  -DBOOST_MPL_AUX_CONFIG_GCC_HPP_INCLUDED "-DBOOST_MPL_CFG_GCC='((__GNUC__ << 8) | __GNUC_MINOR__)'" )
    # force boost to use decltype() for boost::result_of, required to compile without errors (ICC 17)
    set( GT_CXX_MANDATORY_FLAGS ${GT_CXX_MANDATORY_FLAGS}  -DBOOST_RESULT_OF_USE_DECLTYPE )
    # slightly improve performance
    set( GT_CXX_OPTIONAL_FLAGS ${GT_CXX_OPTIONAL_FLAGS}  -qopt-subscript-in-range -qoverride-limits )
    # disable failed vectorization warnings for OpenMP SIMD loops
    set( GT_CXX_OPTIONAL_FLAGS ${GT_CXX_OPTIONAL_FLAGS}  -diag-disable=15518,15552 )
endif()


if(CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
    # Controls preprocessor expansion of macros in Fortran source code.
    set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -eF")
endif()

## performance meters ##
if(GT_ENABLE_PERFORMANCE_METERS)
    set( GT_CXX_BUILDING_FLAGS ${GT_CXX_BUILDING_FLAGS}  -DENABLE_METERS)
endif(GT_ENABLE_PERFORMANCE_METERS)

## precision ##
if(SINGLE_PRECISION)
  if(GT_ENABLE_TARGET_CUDA)
    set(GT_CUDA_BUILDING_FLAGS ${GT_CUDA_BUILDING_FLAGS} -DFLOAT_PRECISION=4 )
  endif()
  set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DFLOAT_PRECISION=4" )
  set( GT_CXX_BUILDING_FLAGS ${GT_CXX_BUILDING_FLAGS} -DFLOAT_PRECISION=4 )
  set( GT_C_BUILDING_FLAGS ${GT_C_BUILDING_FLAGS} -DFLOAT_PRECISION=4 )
  message(STATUS "Computations in single precision")
else()
  if(GT_ENABLE_TARGET_CUDA)
    set(GT_CUDA_BUILDING_FLAGS ${GT_CUDA_BUILDING_FLAGS} -DFLOAT_PRECISION=8)
  endif()
  set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DFLOAT_PRECISION=8" )
  set( GT_CXX_BUILDING_FLAGS ${GT_CXX_BUILDING_FLAGS}  -DFLOAT_PRECISION=8 )
  set( GT_C_BUILDING_FLAGS ${GT_C_BUILDING_FLAGS} -DFLOAT_PRECISION=8 )
  message(STATUS "Computations in double precision")
endif()

## mpi ##
if( GT_USE_MPI )
  set( GT_CXX_MANDATORY_FLAGS ${GT_CXX_MANDATORY_FLAGS}  -D_GCL_MPI_ )
endif()

## caching ##
if( NOT ENABLE_CACHING )
    set( GT_CXX_BUILDING_FLAGS ${GT_CXX_BUILDING_FLAGS} -D__DISABLE_CACHING__ )
endif()


set( GT_CXX_FLAGS ${GT_CXX_BUILDING_FLAGS} ${GT_CXX_OPTIONAL_FLAGS} ${GT_CXX_OPTIMIZATION_FLAGS} ${GT_CXX_MANDATORY_FLAGS} )
string(STRIP "${GT_CXX_FLAGS}" GT_CXX_FLAGS)
set( GT_CUDA_FLAGS ${GT_CUDA_BUILDING_FLAGS} ${GT_CUDA_OPTIONAL_FLAGS} ${GT_CUDA_OPTIMIZATION_FLAGS} ${GT_CUDA_MANDATORY_FLAGS} )
string(STRIP "${GT_CUDA_FLAGS}" GT_CUDA_FLAGS)



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
