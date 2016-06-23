
include_directories( ../../include/communication/low-level/ ) ## Library header files
include_directories( ../../include/communication/high-level ) ## Library header files
### Executables
if( NOT GCL_ONLY )
###   message("${HOST_SPECIFIC_OPTIONS}")
###   message("${CMAKE_CXX_FLAGS}")
###   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HOST_SPECIFIC_OPTIONS}" )
###   message("${CMAKE_CXX_FLAGS}")
  ### L3
  if( GCL_MPI )
    #CPU 2D MPI tests
    foreach(srcfile Halo_Exchange_test_3D proc_grid_3D_test)
      add_executable( ${srcfile} ../../unit_tests/communication/${srcfile}.cpp)
      add_dependencies(${srcfile} gcl)
          set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} ${ADDITIONAL_CXX_FLAGS}")
      target_link_libraries(${srcfile} gcl ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS})
      # TODO communication library tests
      #add_test( test.${srcfile} ${CMAKE_BINARY_DIR}/unit_tests/build/${srcfile})
    endforeach(srcfile)
  endif()

  ### L2
  if( GCL_MPI )
    #CPU 2D MPI tests
    foreach(srcfile test_descriptors_3D test_empty_descriptors_3D boolist ndloops)
      add_executable( ${srcfile} ../../unit_tests/communication/${srcfile}.cpp)
      add_dependencies(${srcfile} gcl)
      set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} ${ADDITIONAL_CXX_FLAGS}")
      target_link_libraries(${srcfile} gcl ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS})
      # TODO communication library tests
      #add_test( test.${srcfile} ${CMAKE_BINARY_DIR}/unit_tests/build/${srcfile})
    endforeach(srcfile)
  endif()
endif()
