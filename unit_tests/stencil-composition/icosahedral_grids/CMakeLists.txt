
if( ENABLE_CXX11 )
  SET(STENCIL_OTHERGRIDS_UNITTEST_SOURCES test_connectivity.cpp test_from_to_mapping.cpp test_make_computation.cpp test_copy_stencil.cpp test_extract_location.cpp 
         ${INC_ALL})

  ### Executables
  #CPU tests
  add_executable(icgrid_stencil_tests ${STENCIL_OTHERGRIDS_UNITTEST_SOURCES})
  gridtools_add_test(tests.stencil_icos_grids ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/icgrid_stencil_tests")

  set_target_properties(icgrid_stencil_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
  target_link_libraries(icgrid_stencil_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
endif( ENABLE_CXX11 )

if(USE_GPU)
    set(CUDA_SOURCES test_copy_stencil.cu)
#test_stencil_on_cells.cu)
    cuda_add_executable(icgrid_stencil_tests_gpu ${CUDA_SOURCES})
    target_link_libraries(icgrid_stencil_tests_gpu ${exe_LIBS})

    gridtools_add_test(tests.icgrid_stencil_tests_gpu ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/icgrid_stencil_tests_gpu")
endif(USE_GPU)
