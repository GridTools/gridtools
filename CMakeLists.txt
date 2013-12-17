cmake_minimum_required( VERSION 2.8 )
project( GRIDTOOLS )
enable_language( CXX )

# include boost
find_package( Boost )
if (Boost_FOUND)
    include_directories( ${Boost_INCLUDE_DIRS} )
endif()

set( USE_GPU "OFF" CACHE BOOL "Compile with GPU support (CUDA)" )

if( "${USE_GPU}" STREQUAL "ON" )
  include(FindCUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_GPU_")
else()
set (CUDA_LIBRARIES "")
endif()

include_directories( include/ ) ## Library header files

set(EXECUTABLE_OUTPUT_PATH "build")
set(LIBRARY_OUTPUT_PATH "lib")


# define a list of all sources
set(
     SOURCES
     "include/arg_type.h"
     "include/array.h"
     "include/axis.h"
     "include/backend0.h"
     "include/backend_naive.h"
     "include/calls.h"
     "include/domain_type.h"
     "include/execution_types.h"
     "include/functor_do_method_lookup_maps.h"
     "include/functor_do_methods.h"
     "include/hasdo.h"
     "include/intermediate.h"
     "include/interval.h"
     "include/layout_map.h"
     "include/level.h"
     "include/loopintervals.h"
     "include/make_stencils.h"
     "include/range.h"
     "include/storage.h"
     "include/yesno.h"
)

add_executable(interface1_notemps_naive   examples/interface1_notemps.cpp ${SOURCES} )
add_executable(interface1_notemps_block   examples/interface1_notemps.cpp ${SOURCES} )
set_target_properties(interface1_notemps_block
        PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")

add_executable(interface1_naive   examples/interface1.cpp ${SOURCES} )
add_executable(interface1_block   examples/interface1.cpp ${SOURCES} )
set_target_properties(interface1_block
        PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")


add_executable(FunctorDoMethodLookupMaps unit_tests/FunctorDoMethodLookupMaps.cpp ${SOURCES})
add_executable(FunctorDoMethods unit_tests/FunctorDoMethods.cpp ${SOURCES})
add_executable(LoopIntervals unit_tests/LoopIntervals.cpp ${SOURCES})

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
endif()
