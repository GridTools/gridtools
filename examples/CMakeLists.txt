## # add_executable(interface1_notemps_naive   interface1_notemps.cpp ${SOURCES} )
## # add_executable(interface1_notemps_block   interface1_notemps.cpp ${SOURCES} )
## # set_target_properties(interface1_notemps_block
## #         PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")

#add_executable(interface2   interface2.cpp ${SOURCES} )


# HERE TESTS THAT TRY ALL THE BACKENDS AND CAN BE COMPILED WITH C++98 AND C++11
foreach(srcfile interface1 boundary-condition laplacian copy_stencil tridiagonal  vertical_advection_dycore vanilla_laplacian extended_4D copy_stencil_parallel )
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
# HERE TESTS THAT TRY ALL THE BACKENDS AND CAN BE COMPILED ***ONLY*** WITH C++11
    add_executable(test_interface   expressions.cpp )
    target_link_libraries(test_interface ${exe_LIBS})
   set_target_properties(test_interface PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

    add_executable(shallow_water  shallow_water.cpp)
    target_link_libraries(shallow_water ${exe_LIBS})
    set_target_properties(shallow_water PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})


##if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
##    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.9)
        add_executable(shallow_water_enhanced  shallow_water_enhanced.cpp)
        target_link_libraries(shallow_water_enhanced gcl ${exe_LIBS})
        set_target_properties(shallow_water_enhanced PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
##    endif()
##endif()


    # add_executable( unstructured unstructured.cpp )
    # target_link_libraries(unstructured ${exe_LIBS})
    # set_target_properties(unstructured PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

endif()

if( NOT ENABLE_CXX11 )
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
#add_executable(basic_laplacian basic_laplacian.cpp)
#target_link_libraries(basic_laplacian ${exe_LIBS})
#add_executable(vanilla_laplacian vanilla_laplacian.cpp)
#target_link_libraries(vanilla_laplacian ${exe_LIBS})

if(CUDA_FOUND)
    if(COMPILE_TO_PTX)
        CUDA_COMPILE_PTX(generated_files interface1.cu copy_stencil.cu tridiagonal.cu boundary-condition.cu)
    else()
        foreach(srcfile interface1 boundary-condition tridiagonal laplacian vertical_advection_dycore copy_stencil copy_stencil_parallel  )
            cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu )
            set_target_properties(${srcfile}_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} )
            target_link_libraries(${srcfile}_cuda gcl ${exe_LIBS})
        endforeach(srcfile)
    endif()

if( NOT ENABLE_CXX11 )
    cuda_add_executable(shallow_water_cxx03_cuda  shallow_water_cxx03.cu)
    target_link_libraries(shallow_water_cxx03_cuda ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
    set_target_properties(shallow_water_cxx03_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

    cuda_add_executable(shallow_water_cxx03_single_storage_cuda  shallow_water_cxx03_single_storage.cu)
    target_link_libraries(shallow_water_cxx03_single_storage_cuda ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
    set_target_properties(shallow_water_cxx03_single_storage_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

    cuda_add_executable(shallow_water_cxx03_cuda_split  shallow_water_cxx03_split.cu)
    target_link_libraries(shallow_water_cxx03_cuda_split ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
    set_target_properties(shallow_water_cxx03_cuda_split PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
else()

    add_executable(shallow_water_enhanced_cuda  shallow_water_enhanced.cu)
    target_link_libraries(shallow_water_enhanced_cuda ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
    set_target_properties(shallow_water_enhanced_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

endif()

endif()

add_subdirectory( communication )
