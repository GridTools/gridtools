include_directories( ../../include/stencil-composition ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/[a-z]*.hpp")

set(RECGRID_TESTS ${INC_ALL})

if ( ENABLE_CXX11 )
    set(RECGRID_TESTS test_local_domain.cpp test_staggered_keyword.cpp test_make_computation.cpp test_cycle_and_swap.cpp test_iterate_domain.cpp test_mss_metafunctions.cpp)

    add_executable(strgrid_stencil_tests ${RECGRID_TESTS})
  # include boost
    include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
      link_directories( ${Boost_LIBRARY_DIRS} )

    set_target_properties(strgrid_stencil_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(strgrid_stencil_tests ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})

    gridtools_add_test(strgrid_stencil_tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/strgrid_stencil_tests")
endif(ENABLE_CXX11)

