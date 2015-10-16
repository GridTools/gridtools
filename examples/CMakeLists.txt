list( APPEND SRCFILES interface1 boundary-condition laplacian copy_stencil positional_copy_stencil tridiagonal vertical_advection_dycore)

if( ENABLE_CXX11)
    list(APPEND SRCFILES  extended_4D  copy_stencil_single_storage)
endif()

set( GTEST_FILES Options.hpp Options.cpp)

set (ADD_MPI_LIBS "")

add_definitions("-DBENCHMARK")

foreach(srcfile IN LISTS SRCFILES )
    add_executable(${srcfile}_naive   ${srcfile}.cpp ${GTEST_FILES})
    add_executable(${srcfile}_block   ${srcfile}.cpp ${GTEST_FILES})
    set_target_properties(${srcfile}_block
      PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
    set_target_properties(${srcfile}_naive PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    set_target_properties(${srcfile}_block PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(${srcfile}_naive ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
    target_link_libraries(${srcfile}_block ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})

    gridtools_add_test( tests.${srcfile}_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/${srcfile}_block" "12 33 61")
    gridtools_add_test( tests.${srcfile}_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/${srcfile}_block" "23 11 43")
    gridtools_add_test( tests.${srcfile}_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/${srcfile}_naive" "12 33 21")
    gridtools_add_test( tests.${srcfile}_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/${srcfile}_naive" "23 11 23")
endforeach(srcfile)


add_executable(vanilla_laplacian  vanilla_laplacian.cpp)
set_target_properties(vanilla_laplacian PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(vanilla_laplacian ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})


if ( ENABLE_CXX11 )
    if( GCL_MPI )
        add_executable(shallow_water_enhanced  shallow_water_enhanced.cpp)
        target_link_libraries(shallow_water_enhanced gcl ${ADD_MPI_LIBS} ${exe_LIBS})
        set_target_properties(shallow_water_enhanced PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

        add_executable(copy_stencil_parallel  copy_stencil_parallel.cpp)
        target_link_libraries(copy_stencil_parallel gcl ${ADD_MPI_LIBS} ${exe_LIBS})
        set_target_properties(copy_stencil_parallel PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
endif()

  endif()

if(CUDA_FOUND)
    if(COMPILE_TO_PTX)
        CUDA_COMPILE_PTX(generated_files interface1.cu copy_stencil.cu tridiagonal.cu boundary-condition.cu)
    else()

        foreach(srcfile IN LISTS SRCFILES MPISRCFILES )
             cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu ${GTEST_FILES})
             target_link_libraries(${srcfile}_cuda ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS} )

             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/${srcfile}_cuda" "77 48 61")
             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/${srcfile}_cuda" "63 55 43")
        endforeach(srcfile)
     endif()

    if( ENABLE_CXX11 )

    cuda_add_executable(shallow_water_enhanced_cuda  shallow_water_enhanced.cu OPTIONS )
    target_link_libraries(shallow_water_enhanced_cuda ${exe_LIBS}  gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})

    cuda_add_executable(copy_stencil_parallel_cuda  copy_stencil_parallel.cu)
    target_link_libraries(copy_stencil_parallel_cuda ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS} )

    endif()
endif()


add_subdirectory( communication )
