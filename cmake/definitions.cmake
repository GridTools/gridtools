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

add_library(gridtools INTERFACE)
add_library(GridTools::gridtools ALIAS gridtools)
# TODO This is a workaround because cmake thinks that clang supports features,
# but it does it wrong because our clang 5.0 RC2 does not match cmakes 5.0
# specification (but 5.0 does)
if (CMAKE_CXX_KNOWN_FEATURES)
    target_compile_features(gridtools INTERFACE cxx_std_11)
endif()
target_include_directories(gridtools
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
      $<INSTALL_INTERFACE:include>
)
include(workaround_icc)
_workaround_icc(gridtools)

set(REQUIRED_BOOST_VERSION 1.58)
find_package( Boost ${REQUIRED_BOOST_VERSION} REQUIRED )
target_link_libraries( gridtools INTERFACE Boost::boost)

if (GT_ENABLE_TARGET_X86 OR GT_ENABLE_TARGET_MC)
    target_link_libraries( gridtools INTERFACE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(gridtools INTERFACE BOOST_PP_VARIADICS=1)
if( GT_ENABLE_TARGET_CUDA )
  target_compile_definitions(gridtools INTERFACE _USE_GPU_)
  if( ${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 8.0 )
      message(FATAL_ERROR "CUDA 7.X or lower is not supported")
  endif()

  # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
  target_compile_options(gridtools INTERFACE
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<STREQUAL:$<TARGET_PROPERTY:CUDA_STANDARD>,14>>:--expt-relaxed-constexpr>)

  if(${GT_CXX_STANDARD} STREQUAL "c++17")
    message(FATAL_ERROR "c++17 is not supported for CUDA compilation")
  endif()

  target_include_directories( gridtools INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
  target_link_libraries( gridtools INTERFACE ${CUDA_CUDART_LIBRARY} )
endif()

# Controls preprocessor expansion of macros in Fortran source code.
# TODO decide where to put this. Probably this should go into fortran bindings
target_compile_options(gridtools INTERFACE $<$<AND:$<CXX_COMPILER_ID:Cray>,$<COMPILE_LANGUAGE:Fortran>>:-eF>)

if( GT_USE_MPI )
    target_compile_definitions(gridtools INTERFACE _GCL_MPI_)
    if( GT_ENABLE_TARGET_CUDA )
      target_compile_definitions(gridtools INTERFACE _GCL_GPU_)
    endif()
endif()

add_library(GridToolsTest INTERFACE)
# NOTE: The CUDA workaround can only be applied to the test because it cannot work
# with generator expressions. Thus, this needs to be redone in the Config.cmake.in.
include(workaround_cuda)
_workaround_cuda(GridToolsTest)
target_link_libraries(GridToolsTest INTERFACE gridtools)
target_compile_definitions(GridToolsTest INTERFACE FUSION_MAX_VECTOR_SIZE=20)
target_compile_definitions(GridToolsTest INTERFACE FUSION_MAX_MAP_SIZE=20)
target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${GT_CUDA_ARCH}>)
if(GT_TESTS_STRUCTURED_GRID)
    target_compile_definitions(GridToolsTest INTERFACE GT_STRUCTURED_GRIDS)
endif()

if( GT_TREAT_WARNINGS_AS_ERROR )
    target_compile_options(GridToolsTest INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror>)
endif()


## clang tools ## TODO (update)
find_package(ClangTools)

# TESTS ONLY
if(GT_ENABLE_TARGET_X86)
  add_library(GridToolsTestX86 INTERFACE)
  target_compile_definitions(GridToolsTestX86 INTERFACE BACKEND_X86)
  target_link_libraries(GridToolsTestX86 INTERFACE GridToolsTest)
  target_compile_options(GridToolsTestX86 INTERFACE -march=native)
endif(GT_ENABLE_TARGET_X86)

## cuda support ##
if( GT_ENABLE_TARGET_CUDA )
  if( GT_TREAT_WARNINGS_AS_ERROR )
     # unfortunately we cannot treat all as warnings, we have to specify each warning; the only supported warning in CUDA8 is cross-execution-space-call
     # CUDA 9 adds deprecated-declarations (activated) and reorder (not activated)
     target_compile_options(GridToolsTest INTERFACE
         $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>
         $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=--warning-as-error>
         $<$<COMPILE_LANGUAGE:CUDA>:-Xnvlink=--warning-as-error>)
     if( ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 9.0 )
         target_compile_options(GridToolsTest INTERFACE
             $<$<COMPILE_LANGUAGE:CUDA>:-Werror=deprecated-declarations>)
     endif()
  endif()

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

# TODO: Move to separate file?
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
if(GT_SINGLE_PRECISION)
  target_compile_definitions(GridToolsTest INTERFACE GT_FLOAT_PRECISION=4)
  message(STATUS "Compile tests in single precision")
else()
  target_compile_definitions(GridToolsTest INTERFACE GT_FLOAT_PRECISION=8)
  message(STATUS "Compile tests in double precision")
endif()

## caching ##
if( NOT GT_TESTS_ENABLE_CACHING )
    # TODO this should be exposed to find_package (GT_ENABLE_CACHING)
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
  set(multi_value_args COMMAND LABELS ENVIRONMENT)
  cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if (NOT ___SCRIPT)
      message(FATAL_ERROR "gridtools_add_test was called without SCRIPT")
  endif()
  if (NOT ___LABELS)
      message(FATAL_ERROR "gridtools_add_test was called without LABELS")
  endif()
  string(REPLACE ";" " " command "${___COMMAND}" )

  file(APPEND ${___SCRIPT} "echo ${command}\n")
  file(APPEND ${___SCRIPT} "${command}\n")
  file(APPEND ${___SCRIPT} "res=$((res || $? ))\n")
  add_to_test_manifest("${___NAME} ${command}")

  if (NOT ${TEST_USE_WRAPPERS_FOR_ALL_TESTS})
    add_test(NAME ${___NAME} COMMAND ${___COMMAND})
  else()
    add_test(
        NAME ${___NAME}
        COMMAND  ${MPITEST_EXECUTABLE} ${MPITEST_NUMPROC_FLAG} 1 ${MPITEST_PREFLAGS} ${___COMMAND} ${MPITEST_POSTFLAGS}
        )
  endif()
  if (${___ENVIRONMENT})
      set_tests_properties(${___NAME} PROPERTIES ENVIRONMENT "${___ENVIRONMENT}")
  endif()
  set_tests_properties(${___NAME} PROPERTIES LABELS "${___LABELS}")
endfunction(gridtools_add_test)

file(WRITE ${TEST_MPI_SCRIPT} "res=0\n")
file(GENERATE OUTPUT ${TEST_MPI_SCRIPT} INPUT ${TEST_MPI_SCRIPT})

file(WRITE ${TEST_CUDA_MPI_SCRIPT} "res=0\n")
file(GENERATE OUTPUT ${TEST_CUDA_MPI_SCRIPT} INPUT ${TEST_CUDA_MPI_SCRIPT})

## test script generator for MPI tests ##
function(gridtools_add_mpi_test)
  set(options)
  set(one_value_args NAME SCRIPT NPROC )
  set(multi_value_args COMMAND LABELS ENVIRONMENT)
  cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if (NOT ___SCRIPT)
      message(FATAL_ERROR "gridtools_add_mpi_test was called without SCRIPT")
  endif()
  if (NOT ___NPROC)
      message(FATAL_ERROR "gridtools_add_mpi_test was called without NPROC")
  endif()
  if (NOT ___LABELS)
      message(FATAL_ERROR "gridtools_add_mpi_test was called without LABELS")
  endif()
  string(REPLACE ";" " " command "${___COMMAND}" )

  file(APPEND ${___SCRIPT} "echo \$LAUNCH_MPI_TEST ${command}\n")
  file(APPEND ${___SCRIPT} "\$LAUNCH_MPI_TEST ${command}\n")
  file(APPEND ${___SCRIPT} "res=$((res || $? ))\n")
  add_to_test_manifest("${___NAME} ${command}")
  # Note: We use MPITEST_ instead of MPIEXEC_ because our own MPI_TEST_-variables are slurm-aware
  add_test(
      NAME ${___NAME}
      COMMAND  ${MPITEST_EXECUTABLE} ${MPITEST_NUMPROC_FLAG} ${___NPROC} ${MPITEST_PREFLAGS} ${___COMMAND} ${MPITEST_POSTFLAGS}
      )
  if (___ENVIRONMENT)
      string(REPLACE " " ";" environment "${___ENVIRONMENT}" )
      set_tests_properties(${___NAME} PROPERTIES ENVIRONMENT "${environment}")
  endif()
  set_tests_properties(${___NAME} PROPERTIES LABELS "${___LABELS}")
  set_tests_properties(${___NAME} PROPERTIES PROCESSORS ${___NPROC})
endfunction()

