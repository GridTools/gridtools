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

add_library(GridTools INTERFACE)
target_include_directories(GridTools
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
      $<INSTALL_INTERFACE:include>
)

find_package( Boost 1.58 REQUIRED )
target_link_libraries( GridTools INTERFACE Boost::boost)

find_package( OpenMP REQUIRED )
target_link_libraries( GridTools INTERFACE OpenMP::OpenMP_CXX)

set(THREADS_PREFER_PTHREAD_FLAG ON) #this is required because gtest uses it
find_package( Threads REQUIRED )
target_link_libraries( GridTools INTERFACE Threads::Threads)
include(workaround_threads)
_fix_threads_flags()

target_compile_definitions(GridTools INTERFACE SUPPRESS_MESSAGES)
target_compile_definitions(GridTools INTERFACE BOOST_PP_VARIADICS=1)
if(STRUCTURED_GRIDS)
    target_compile_definitions(GridTools INTERFACE STRUCTURED_GRIDS)
endif()
if( GT_ENABLE_TARGET_CUDA )
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR "${CMAKE_CUDA_COMPILER_VERSION}")
  target_compile_definitions(GridTools INTERFACE GT_CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR})
  target_compile_definitions(GridTools INTERFACE GT_CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})
  target_compile_definitions(GridTools INTERFACE _USE_GPU_)
  if( "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS "8.0" )
      message(FATAL_ERROR "CUDA 7.X or lower is not supported")
  endif()
  target_compile_options(GridTools INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_ARCH}>)

  # workaround for boost::optional with CUDA9.2
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9.2")
      target_compile_definitions(GridTools INTERFACE BOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL)
      target_compile_definitions(GridTools INTERFACE BOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE)
  endif()

  # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
  target_compile_options(GridTools INTERFACE
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<EQUAL:$<TARGET_PROPERTY:CUDA_STANDARD>,14>>:--expt-relaxed-constexpr>)

  if(${GT_CXX_STANDARD} STREQUAL "c++17")
    message(FATAL_ERROR "c++17 is not supported for CUDA compilation")
  endif()

  target_include_directories( GridTools INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

  # this is only needed to get CUDA_CUDART_LIBRARY, please do not use other variables from here!
  # Find a better solution for this (consider https://gitlab.kitware.com/cmake/cmake/issues/17816)
  find_package(CUDA REQUIRED)
  target_link_libraries( GridTools INTERFACE ${CUDA_CUDART_LIBRARY} )
endif()
# TODO check with ICC 18
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # fix buggy Boost MPL config for Intel compiler (last confirmed with Boost 1.65 and ICC 17)
    # otherwise we run into this issue: https://software.intel.com/en-us/forums/intel-c-compiler/topic/516083
    target_compile_definitions(GridTools INTERFACE $<$<CXX_COMPILER_ID:Intel>:BOOST_MPL_AUX_CONFIG_GCC_HPP_INCLUDED>)
    target_compile_definitions(GridTools INTERFACE "$<$<CXX_COMPILER_ID:Intel>:BOOST_MPL_CFG_GCC='((__GNUC__ << 8) | __GNUC_MINOR__)'>" )

    # force boost to use decltype() for boost::result_of, required to compile without errors (ICC 17)
    target_compile_definitions(GridTools INTERFACE $<$<CXX_COMPILER_ID:Intel>:BOOST_RESULT_OF_USE_DECLTYPE>)
endif()
#TODO decide where to put this
if(CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
    # Controls preprocessor expansion of macros in Fortran source code.
    target_compile_options(GridTools INTERFACE $<AND:$<CXX_COMPILER_ID:Cray>,$<COMPILER_LANGUAGE:Fortran>>:-eF>)
endif()
if( GT_USE_MPI )
    target_compile_definitions(GridTools INTERFACE _GCL_MPI_)
    if( GT_ENABLE_TARGET_CUDA )
      target_compile_definitions(GridTools INTERFACE _GCL_GPU_)
    endif()
endif()

add_library(GridToolsTest INTERFACE)
target_link_libraries(GridToolsTest INTERFACE GridTools)
target_compile_definitions(GridToolsTest INTERFACE FUSION_MAX_VECTOR_SIZE=20)
target_compile_definitions(GridToolsTest INTERFACE FUSION_MAX_MAP_SIZE=20)
if(NOT GT_ENABLE_TARGET_CUDA AND NOT GT_ENABLE_TARGET_MC)
    target_compile_options(GridToolsTest -march=native)
endif()

if( GT_TREAT_WARNINGS_AS_ERROR )
    target_compile_options(GridToolsTest INTERFACE -Werror)
endif()


## clang tools ## TODO (update)
find_package(ClangTools)

# TESTS ONLY
if(GT_ENABLE_TARGET_X86)
  add_library(GridToolsTestX86 INTERFACE)
  target_compile_definitions(GridToolsTestX86 INTERFACE BACKEND_X86)
  target_link_libraries(GridToolsTestX86 INTERFACE GridToolsTest)
endif(GT_ENABLE_TARGET_X86)

## cuda support ##
if( GT_ENABLE_TARGET_CUDA )
  if( GT_TREAT_WARNINGS_AS_ERROR )
     #unfortunately we cannot treat all as warnings, we have to specify each warning; the only supported warning in CUDA8 is cross-execution-space-call
     target_compile_options(GridToolsTest INTERFACE -Werror cross-execution-space-call -Xptxas --warning-as-error -nvlink-options --warning-as-error)
  endif()

  # suppress because of a warning coming from gtest.h
  target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=code_is_unreachable>)
  if( ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 9.0 )
    # suppress because of boost::fusion::vector ctor
    target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored>)
  endif()

  add_library(GridToolsTestCUDA INTERFACE)
  target_compile_definitions(GridToolsTestCUDA INTERFACE BACKEND_CUDA)
  target_link_libraries(GridToolsTestCUDA INTERFACE GridToolsTest)
endif()

if( GT_ENABLE_TARGET_MC )
  add_library(GridToolsTestMC INTERFACE)
  target_compile_definitions(GridToolsTestMC INTERFACE BACKEND_MC)
  target_link_libraries(GridToolsTestMC INTERFACE GridToolsTest)
endif( GT_ENABLE_TARGET_MC )

## clang ##
if((CUDA_HOST_COMPILER MATCHES "(C|c?)lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "(C|c?)lang"))
    # set( GT_CXX_HOST_ONLY_FLAGS ${GT_CXX_HOST_ONLY_FLAGS}  -ftemplate-depth-1024 )
    # disable failed vectorization warnings for OpenMP SIMD loops
    # set( GT_CXX_HOST_ONLY_FLAGS ${GT_CXX_HOST_ONLY_FLAGS}  -Wno-pass-failed )
endif()
# TODO check with ICC 18
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # TODO add those flags to documentation (slightly improve performance)
    target_compile_options(GridToolsTest INTERFACE -qopt-subscript-in-range -qoverride-limits)
    # disable failed vectorization warnings for OpenMP SIMD loops
    target_compile_options(GridToolsTest INTERFACE -diag-disable=15518,15552)
endif()

## performance meters ##
if(GT_ENABLE_PERFORMANCE_METERS)
    target_compile_definitions(GridToolsTest INTERFACE ENABLE_METERS)
endif(GT_ENABLE_PERFORMANCE_METERS)

## precision ##
if(SINGLE_PRECISION)
  # TODO move to GridToolsTest
  target_compile_definitions(GridTools INTERFACE FLOAT_PRECISION=4)
  message(STATUS "Compile tests in single precision")
else()
  # TODO move to GridToolsTest
  target_compile_definitions(GridTools INTERFACE FLOAT_PRECISION=8)
  message(STATUS "Compile tests in double precision")
endif()

## caching ##
if( NOT GT_TESTS_ENABLE_CACHING )
    # TODO this should not be a cached option (it might be an option for tests + an option that is set before
    # find_package). Note that we only attach it to tests
    target_compile_definitions(GridToolsTest __DISABLE_CACHING__)
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
file(GENERATE OUTPUT ${TEST_SCRIPT} INPUT ${TEST_SCRIPT})
file(APPEND ${TEST_SCRIPT} "hostname\n")
file(APPEND ${TEST_SCRIPT} "res=0\n")
function(gridtools_add_test)
  set(options)
  set(one_value_args NAME SCRIPT )
  set(multi_value_args COMMAND LABELS)
  cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  string(REPLACE ";" " " command "${___COMMAND}" )

  file(APPEND ${___SCRIPT} "echo ${command}\n")
  file(APPEND ${___SCRIPT} "${command}\n")
  file(APPEND ${___SCRIPT} "res=$((res || $? ))\n")
  add_to_test_manifest("${___NAME} ${command}")
  add_test(NAME ${___NAME} COMMAND ${___COMMAND})
  set_tests_properties(${___NAME} PROPERTIES LABELS "${___LABELS}")
endfunction(gridtools_add_test)

## test script generator for MPI tests ##
file(WRITE ${TEST_MPI_SCRIPT} "res=0\n")
file(GENERATE OUTPUT ${TEST_MPI_SCRIPT} INPUT ${TEST_MPI_SCRIPT})
function(gridtools_add_mpi_test)
  set(options)
  set(one_value_args NAME )
  set(multi_value_args COMMAND LABELS)
  cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  string(REPLACE ";" " " command "${___COMMAND}" )

  file(APPEND ${TEST_MPI_SCRIPT} "echo \$LAUNCH_MPI_TEST ${command}\n")
  file(APPEND ${TEST_MPI_SCRIPT} "\$LAUNCH_MPI_TEST ${command}\n")
  file(APPEND ${TEST_MPI_SCRIPT} "res=$((res || $? ))\n")
  add_to_test_manifest("${___NAME} ${command}")
  add_test(
      NAME ${___NAME}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ${___COMMAND} ${MPIEXEC_POSTFLAGS}
      )
  set_tests_properties(${___NAME} PROPERTIES LABELS "${___LABELS}")
  if (MPIEXEC_MAX_NUMPROCS)
      set_tests_properties(${___NAME} PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
  endif()
endfunction(gridtools_add_mpi_test)

file(WRITE ${TEST_CUDA_MPI_SCRIPT} "res=0\n")
file(GENERATE OUTPUT ${TEST_CUDA_MPI_SCRIPT} INPUT ${TEST_CUDA_MPI_SCRIPT})
function(gridtools_add_cuda_mpi_test )
  set(options)
  set(one_value_args NAME )
  set(multi_value_args COMMAND LABELS)
  cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  string(REPLACE ";" " " command "${___COMMAND}" )

  file(APPEND ${TEST_CUDA_MPI_SCRIPT} "echo \$LAUNCH_MPI_TEST ${command}\n")
  file(APPEND ${TEST_CUDA_MPI_SCRIPT} "\$LAUNCH_MPI_TEST ${command}\n")
  file(APPEND ${TEST_CUDA_MPI_SCRIPT} "res=$((res || $? ))\n")
  add_to_test_manifest(${___NAME} ${___COMMAND})
  add_test(
      NAME ${___NAME}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ${___COMMAND} ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${___NAME} PROPERTIES LABELS "${___LABELS}")
  if (MPIEXEC_MAX_NUMPROCS)
      set_tests_properties(${___NAME} PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
  endif()
endfunction(gridtools_add_cuda_mpi_test)
