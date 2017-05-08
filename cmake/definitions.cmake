# set cxx11 flag
set(ENABLE_CXX11)

## set boost fusion sizes ##
add_definitions(-DFUSION_MAX_VECTOR_SIZE=${BOOST_FUSION_MAX_SIZE})
add_definitions(-DFUSION_MAX_MAP_SIZE=${BOOST_FUSION_MAX_SIZE})

## get boost ##
message(STATUS "try to find boost in: " ${BOOST_ROOT})
set( BOOST_DIR "${BOOST_ROOT}")
set( Boost_DIR ${BOOST_ROOT} )
find_package( Boost 1.58.0 )

if(NOT Boost_FOUND)
    message(WARNING "Boost library could not be found: using fallback solution")
    include_directories("${BOOST_ROOT}/include")
    set(exe_LIBS, "${BOOST_ROOT}/lib" "${exe_LIBS}")
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(exe_LIBS "${Boost_LIBRARIES}" "${exe_LIBS}")
endif()

## gnu coverage flag ##
if(GNU_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lgcov" )
    message (STATUS "Building executables for coverage tests")
endif()

## gnu profiling information ##
if(GNU_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs")
    message (STATUS "Building profiled executables")
endif()

## cuda support ##
if( USE_GPU )
  message(STATUS "Using GPU")
  find_package(CUDA REQUIRED)
  string(REPLACE "." "" CUDA_VERSION ${CUDA_VERSION})
  add_definitions(-DCUDA_VERSION=${CUDA_VERSION})
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  if( ${CUDA_VERSION} VERSION_GREATER "60")
      set(CUDA_NVCC_FLAGS "--std=c++11" "--relaxed-constexpr" "${CUDA_NVCC_FLAGS}")
  else()
      message(STATUS "CUDA 6.0 or lower does not support C++11 (disabling)")
      set(CUDA_NVCC_FLAGS "-DCXX11_DISABLE" "${CUDA_NVCC_FLAGS}")
      set(ENABLE_CXX11 "OFF" )
  endif()
  set( CUDA_ARCH "sm_35" CACHE STRING "Compute capability for CUDA" )
  set( CUDA_SEPARABLE_COMPILATION ON)
  
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  
  set(exe_LIBS "${CUDA_CUDART_LIBRARY}" "${exe_LIBS}" )
  set(CUDA_SEPARABLE_COMPILATION OFF)
  # adding the additional nvcc flags
  set(CUDA_NVCC_FLAGS_MINSIZEREL "${CUDA_NVCC_FLAGS_MINSIZEREL}" "-Os" "-DNDEBUG")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE}" "-O3" "-DNDEBUG")
  set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "${CUDA_NVCC_FLAGS_RELWITHDEBINFO}" "-O2" "-g" "-DNDEBUG")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-arch=${CUDA_ARCH}" "-Xcudafe" "--diag_suppress=dupl_calling_convention")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcudafe" "--diag_suppress=code_is_unreachable" "-Xcudafe")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--diag_suppress=implicit_return_from_non_void_function" "-Xcudafe")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--diag_suppress=calling_convention_not_allowed" "-Xcudafe") 
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--diag_suppress=conflicting_calling_conventions")
else()
  set (CUDA_LIBRARIES "")
  set( CUDA_CXX11 " ")
endif()

set(exe_LIBS ${GTEST_LIBRARIES} -lpthread ${exe_LIBS})

## mpi ##
if( USE_MPI )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GCL_MPI_")
  if( USE_MPI_COMPILER )
    find_package(MPI REQUIRED)
    include(CMakeForceCompiler)
    cmake_force_cxx_compiler(mpicxx "MPI C++ Compiler")
  endif()
endif()

## test script generator ## 
file(WRITE ${TEST_SCRIPT} "#!/bin/sh\n")
file(APPEND ${TEST_SCRIPT} "res=0\n")
function(gridtools_add_test test_name test_script test_exec)
  file(APPEND ${test_script} "${test_exec}" " ${ARGN}" "\n")
  file(APPEND ${test_script} "res=$((res || $? ))\n")
endfunction(gridtools_add_test)


