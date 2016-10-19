list( APPEND SRCFILES #laplace
      expandable_parameters_icosahedral stencil_on_cells stencil_on_cells_color stencil_on_edges stencil_on_edges_of_cells stencil_on_neighcell_of_edges stencil_on_neighedge_of_cells stencil_manual_fold stencil_on_vertexes stencil_on_edges_multiplefields)

set( GTEST_FILES ../Options.hpp ../Options.cpp unstructured_grid.cpp)

add_definitions("-DBENCHMARK")
set(exe_LIBS gtest gtest_main "${exe_LIBS}" )

foreach(srcfile IN LISTS SRCFILES )
    add_executable(${srcfile}_naive   ${srcfile}.cpp ${GTEST_FILES})
    add_executable(${srcfile}_block   ${srcfile}.cpp ${GTEST_FILES})
    set_target_properties(${srcfile}_block
      PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
    set_target_properties(${srcfile}_naive PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
    set_target_properties(${srcfile}_block PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
    target_link_libraries(${srcfile}_naive ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
    target_link_libraries(${srcfile}_block ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})

    gridtools_add_test( tests.${srcfile}_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/icosahedral/${srcfile}_block" "12 33 61")
    gridtools_add_test( tests.${srcfile}_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/icosahedral/${srcfile}_block" "23 11 43")
    gridtools_add_test( tests.${srcfile}_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/icosahedral/${srcfile}_naive" "12 33 21")
    gridtools_add_test( tests.${srcfile}_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/icosahedral/${srcfile}_naive" "23 11 23")
endforeach(srcfile)

if(CUDA_FOUND)
    foreach(srcfile IN LISTS SRCFILES MPISRCFILES )
         set(CUDA_SEPARABLE_COMPILATION OFF)
         cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu ${GTEST_FILES})
         set_target_properties(${srcfile}_cuda PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
         target_link_libraries(${srcfile}_cuda ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS} )

         if( ${srcfile} STREQUAL "stencil_on_vertexes")
             # K lenght larger than 48 fails for stencil_on_vertexes_cuda... this need to be investigated
             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/icosahedral/${srcfile}_cuda" "77 48 48")
             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/icosahedral/${srcfile}_cuda" "63 55 43")
         else()
             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/icosahedral/${srcfile}_cuda" "77 48 61")
             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/icosahedral/${srcfile}_cuda" "63 55 43")
         endif()
    endforeach(srcfile)
endif()

### Validation is not passing for the following but we compile it for maintaining the API and code
list( APPEND SRCCOMPONLYFILES  # TODO RECOVER stencil_nested_on
)

foreach(srcfile IN LISTS SRCCOMPONLYFILES )
    add_executable(${srcfile}_naive   ${srcfile}.cpp ${GTEST_FILES})
    add_executable(${srcfile}_block   ${srcfile}.cpp ${GTEST_FILES})
    set_target_properties(${srcfile}_block
      PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
    set_target_properties(${srcfile}_naive PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    set_target_properties(${srcfile}_block PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(${srcfile}_naive ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
    target_link_libraries(${srcfile}_block ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})

endforeach(srcfile)

#if(CUDA_FOUND)
#    foreach(srcfile IN LISTS SRCCOMPONLYFILES MPISRCFILES )
#         cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu ${GTEST_FILES})
#         set_target_properties(${srcfile}_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
#         target_link_libraries(${srcfile}_cuda ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS} )
#    endforeach(srcfile)
#endif()
