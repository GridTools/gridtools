cmake_minimum_required( VERSION 2.8 )
project( GRIDTOOLS )
enable_language( CXX )

# include boost
find_package( Boost )
if (Boost_FOUND)
    include_directories( ${Boost_INCLUDE_DIRS} )
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


