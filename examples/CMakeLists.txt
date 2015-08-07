## # add_executable(interface1_notemps_naive   interface1_notemps.cpp ${SOURCES} )
## # add_executable(interface1_notemps_block   interface1_notemps.cpp ${SOURCES} )
## # set_target_properties(interface1_notemps_block
## #         PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")

#add_executable(interface2   interface2.cpp ${SOURCES} )


# HERE TESTS THAT TRY ALL THE BACKENDS AND CAN BE COMPILED WITH C++98 AND C++11

#TODOCOSUNA RECOVER FILES
#list( APPEND SRCFILES interface1 boundary-condition laplacian copy_stencil positional_copy_stencil tridiagonal  vertical_advection_dycore vanilla_laplacian extended_4D )
list( APPEND SRCFILES interface1 boundary-condition laplacian copy_stencil positional_copy_stencil tridiagonal  vertical_advection_dycore vanilla_laplacian extended_4D)

if( GCL_MPI )
    list(APPEND SRCFILES copy_stencil_parallel)
endif()

add_definitions("-DBENCHMARK")

foreach(srcfile IN LISTS SRCFILES )
    add_executable(${srcfile}_naive   ${srcfile}.cpp )
    add_executable(${srcfile}_block   ${srcfile}.cpp )
    set_target_properties(${srcfile}_block
      PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
    set_target_properties(${srcfile}_naive PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    set_target_properties(${srcfile}_block PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(${srcfile}_naive ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
    target_link_libraries(${srcfile}_block ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
endforeach(srcfile)

if ( ENABLE_CXX11 )
    if( GCL_MPI )
        # HERE TESTS THAT TRY ALL THE BACKENDS AND CAN BE COMPILED ***ONLY*** WITH C++11
        add_executable(shallow_water  shallow_water.cpp)
        target_link_libraries(shallow_water ${exe_LIBS})
        set_target_properties(shallow_water PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})


        add_executable(shallow_water_enhanced  shallow_water_enhanced.cpp)
         target_link_libraries(shallow_water_enhanced gcl ${exe_LIBS})
         set_target_properties(shallow_water_enhanced PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    endif()

    # add_executable( unstructured unstructured.cpp )
    # target_link_libraries(unstructured ${exe_LIBS})
    # set_target_properties(unstructured PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

  endif()

if( NOT ENABLE_CXX11 )
    if( GCL_MPI )
        add_executable(shallow_water_cxx03  shallow_water_cxx03.cpp)
        target_link_libraries(shallow_water_cxx03 ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
        set_target_properties(shallow_water_cxx03 PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

        add_executable(shallow_water_cxx03_single_storage  shallow_water_cxx03_single_storage.cpp)
        target_link_libraries(shallow_water_cxx03_single_storage ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
        set_target_properties(shallow_water_cxx03_single_storage PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

        add_executable(shallow_water_cxx03_split  shallow_water_cxx03_split.cpp)
        target_link_libraries(shallow_water_cxx03_split ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
        set_target_properties(shallow_water_cxx03_split PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    endif()
endif()

if(CUDA_FOUND)
    if(COMPILE_TO_PTX)
        CUDA_COMPILE_PTX(generated_files interface1.cu copy_stencil.cu tridiagonal.cu boundary-condition.cu)
    else()

        if( GCL_MPI )
             cuda_add_executable( copy_stencil_parallel_cuda copy_stencil_parallel.cu )
             target_link_libraries(copy_stencil_parallel_cuda gcl ${exe_LIBS})
        endif()

        foreach(srcfile interface1 boundary-condition tridiagonal laplacian vertical_advection_dycore copy_stencil )
             cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu )

             target_link_libraries(${srcfile}_cuda gcl ${exe_LIBS})
         endforeach(srcfile)
     endif()

    if( NOT ENABLE_CXX11 )
        if( GCL_MPI )
            cuda_add_executable(shallow_water_cxx03_cuda  shallow_water_cxx03.cu)
            target_link_libraries(shallow_water_cxx03_cuda ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

            cuda_add_executable(shallow_water_cxx03_single_storage_cuda  shallow_water_cxx03_single_storage.cu OPTIONS ${ADDITIONAL_CXX_FLAGS})
            target_link_libraries(shallow_water_cxx03_single_storage_cuda ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

            cuda_add_executable(shallow_water_cxx03_cuda_split  shallow_water_cxx03_split.cu OPTIONS ${ADDITIONAL_CXX_FLAGS})
            target_link_libraries(shallow_water_cxx03_cuda_split ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
        endif()
    else()

#      cuda_add_executable(shallow_water_enhanced_cuda  shallow_water_enhanced.cu OPTIONS )
#      target_link_libraries(shallow_water_enhanced_cuda ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

    endif()


    foreach(srcfile positional_copy_stencil )
        cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu OPTIONS -DNDEBUG )
        target_link_libraries(${srcfile}_cuda ${exe_LIBS})
    endforeach(srcfile)
endif()


add_subdirectory( communication )
