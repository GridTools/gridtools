cmake_minimum_required( VERSION 2.8 )
project( GRIDTOOLS )
enable_language( CXX )

enable_testing()

# include boost
set( USE_GPU "OFF" CACHE BOOL "Compile with GPU support (CUDA)" )
set( GPU_ENABLED_FUSION "../fusion/include" CACHE STRING "Path to...")

if( "${USE_GPU}" STREQUAL "ON" )
  include(FindCUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_GPU_")
else()
set (CUDA_LIBRARIES "")
endif()

include_directories( include/ ) ## Library header files
include_directories( ${GPU_ENABLED_FUSION} ) ## Library header files
include_directories( $ENV{GTEST_ROOT}/include )

find_package( Boost )
if (Boost_FOUND)
    include_directories( ${Boost_INCLUDE_DIRS} )
endif()

set(EXECUTABLE_OUTPUT_PATH "build")
set(LIBRARY_OUTPUT_PATH "lib")

# define a list of all sources
set(
     SOURCES
     "include/arg_type.h"
	"include/array.h"
	"include/axis.h"
	"include/backend0.h"
	"include/backend_block.h"
	"include/backend_cuda.h"
	"include/backend_naive.h"
	"include/base_storage.h"
	"include/basic_token_execution.h"
	"include/basic_utils.h"
	"include/cuda_storage.h"
	"include/defs.h"
	"include/domain_type.h"
	"include/domain_type_impl.h"
	"include/execution_types.h"
	"include/functor_do_method_lookup_maps.h"
	"include/functor_do_methods.h"
	"include/gpu_clone.h"
	"include/gridtools.h"
	"include/gt_assert.h"
	"include/hasdo.h"
	"include/host_device.h"
	"include/hybrid_pointer.h"
	"include/intermediate.h"
	"include/interval.h"
	"include/layout_map.h"
	"include/level.h"
	"include/local_domain.h"
	"include/loopintervals.h"
	"include/make_stencils.h"
	"include/range.h"
	"include/storage.h"
	"include/yesno.h"
	"include/gt_for_each/for_each.hpp"
	"include/gt_for_each/unwrap.hpp"
	"include/gt_for_each/value_init.hpp"
)

## # add_executable(interface1_notemps_naive   examples/interface1_notemps.cpp ${SOURCES} )
## # add_executable(interface1_notemps_block   examples/interface1_notemps.cpp ${SOURCES} )
## # set_target_properties(interface1_notemps_block
## #         PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")

#add_executable(interface2   examples/interface2.cpp ${SOURCES} )

add_executable(interface1_naive   examples/interface1.cpp ${SOURCES} )
add_executable(interface1_block   examples/interface1.cpp ${SOURCES} )
set_target_properties(interface1_block
        PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")


add_executable(FunctorDoMethodLookupMaps unit_tests/FunctorDoMethodLookupMaps.cpp ${SOURCES})
add_executable(FunctorDoMethods unit_tests/FunctorDoMethods.cpp ${SOURCES})
add_executable(LoopIntervals unit_tests/LoopIntervals.cpp ${SOURCES})

set (test_gtest  $ENV{GTEST_ROOT})
if ("${test_gtest}" STREQUAL "")
   message("GTEST_ROOT not defined: tests targets not compiled")
else()
    add_executable(tests unit_tests/tests.cpp unit_tests/test_domain_indices.cpp )
    target_link_libraries( tests $ENV{GTEST_ROOT}/libgtest.a $ENV{GTEST_ROOT}/libgtest_main.a -lpthread )
endif()

add_executable(
        test-independent
        unit_tests/test-independent.cpp
        ${SOURCES}
)
set_target_properties(test-independent
        PROPERTIES COMPILE_FLAGS "-std=c++11")

if( "${USE_GPU}" STREQUAL "ON" )
    cuda_add_executable( interface1_cuda examples/interface1.cu )
    #add_dependencies(  )
    #target_link_libraries( interface1_cuda ${CUDA_MPI_LIB} )
    cuda_add_executable( cloning_derived unit_tests/cloning_derived.cu )

    if (test_gtest STREQUAL "")
    else()
        cuda_add_executable( tests_gpu unit_tests/tests.cu unit_tests/test_domain.cu unit_tests/test_cuda_storage.cu unit_tests/test_hybrid_pointer.cu unit_tests/gpu_clone.cu unit_tests/cloningstuff.cu )
        target_link_libraries( tests_gpu $ENV{GTEST_ROOT}/libgtest.a $ENV{GTEST_ROOT}/libgtest_main.a )
    endif()

endif()
