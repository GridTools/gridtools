include_directories( ../../include/stencil-composition ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/[a-z]*.hpp")

set(RECGRID_TESTS ${INC_ALL})

if ( ENABLE_CXX11 )
    set(RECGRID_TESTS test_local_domain.cpp test_staggered_keyword.cpp)

    add_executable(recgrid_stencil_tests ${RECGRID_TESTS})
  # include boost
    include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
      link_directories( ${Boost_LIBRARY_DIRS} )

    set_target_properties(recgrid_stencil_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(recgrid_stencil_tests ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

    gridtools_add_test(recgrid_stencil_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/recgrid_stencil_tests")
endif(ENABLE_CXX11)

