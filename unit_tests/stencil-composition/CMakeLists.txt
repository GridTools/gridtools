
include_directories( ../../include/stencil-composition ) ## Library header files

file(GLOB_RECURSE INC_ALL "../../include/stencil-composition/*.hpp")

set(STENCIL_COMPOSITION_UNITTEST_SOURCES test_accessor.cpp ${INC_ALL})
if ( ENABLE_CXX11 )
  if( RECTANGULAR_GRIDS )
    set(STENCIL_COMPOSITION_UNITTEST_SOURCES ${STENCIL_COMPOSITION_UNITTEST_SOURCES} test_iterate_domain.cpp test_mss_metafunctions.cpp test_make_computation)
  endif()
endif( ENABLE_CXX11 )


### Executables
#CPU tests
add_executable(stencil_composition_tests ${STENCIL_COMPOSITION_UNITTEST_SOURCES})
gridtools_add_test(tests.stencil_composition ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/stencil_composition_tests")

set_target_properties(stencil_composition_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(stencil_composition_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

add_subdirectory(caches)
if (NOT RECTANGULAR_GRIDS)
  add_subdirectory(other_grids)
endif()
