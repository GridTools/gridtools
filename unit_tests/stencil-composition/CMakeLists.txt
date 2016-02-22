
include_directories( ../../include/stencil-composition ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/[a-z]*.hpp")

SET(STENCIL_COMPOSITION_UNITTEST_SOURCES test_accessor.cpp  test_conditionals.cpp ${INC_ALL})
if ( ENABLE_CXX11 )
  SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${STENCIL_COMPOSITION_UNITTEST_SOURCES} test-assign-placeholders.cpp  test_conditional_switches.cpp)
endif( ENABLE_CXX11 )


### Executables
#CPU tests
add_executable(stencil_composition_tests ${STENCIL_COMPOSITION_UNITTEST_SOURCES})
gridtools_add_test(tests.stencil_composition ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/stencil_composition_tests")

set_target_properties(stencil_composition_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(stencil_composition_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

add_executable(test_conditionals test_conditionals.cpp)
set_target_properties(test_conditionals PROPERTIES COMPILE_FLAGS "${ADDITIONAL_CXX_FLAGS}" )
target_link_libraries(test_conditionals ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

if (ENABLE_CXX11)
add_executable(test_conditional_switches test_conditional_switches.cpp)
set_target_properties(test_conditional_switches PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(test_conditional_switches ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
endif(ENABLE_CXX11)

add_subdirectory(caches)
if (NOT STRUCTURED_GRIDS)
  add_subdirectory(icosahedral_grids)
else()
  add_subdirectory(structured_grids)
endif()
