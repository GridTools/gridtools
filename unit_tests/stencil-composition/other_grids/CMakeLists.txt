
include_directories( ../../../include/stencil-composition ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../../nclude/stencil-composition/*.hpp")

if( ENABLE_CXX11 )
  SET(STENCIL_OTHERGRIDS_UNITTEST_SOURCES test_from_to_mapping.cpp test_make_computation.cpp test_copy_stencil.cpp unstructured_grid.cpp test_extract_location.cpp 
         test_stencil_on_cells.cpp 
         ${INC_ALL})
endif( ENABLE_CXX11 )

### Executables
#CPU tests
add_executable(stencil_other_grids_tests ${STENCIL_OTHERGRIDS_UNITTEST_SOURCES})
gridtools_add_test(tests.stencil_other_grids ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/stencil_other_grids_tests")

set_target_properties(stencil_other_grids_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(stencil_other_grids_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

