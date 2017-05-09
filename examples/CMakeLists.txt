if(NOT STRUCTURED_GRIDS)
    add_subdirectory(icosahedral)
else()

  file(GLOB_RECURSE INC_ALL_ "[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )
  file(GLOB_RECURSE INC_ALL_ "../include/[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )

  list( APPEND SRCFILES interface1 simple_hori_diff boundary-condition laplacian copy_stencil positional_copy_stencil tridiagonal vertical_advection_dycore alignment )
  list( APPEND X86_SRCFILES reduction)

  if( ENABLE_CXX11)
    list(APPEND SRCFILES  extended_4D  copy_stencil_single_storage  expandable_parameters  expandable_parameters_single_kernel expandable_parameters_reassign_domain advection_pdbott_prepare_tracers)
  endif()

  set( GTEST_FILES Options.hpp Options.cpp)

  add_definitions("-DBENCHMARK")

  foreach(srcfile IN LISTS SRCFILES X86_SRCFILES)
    add_executable(${srcfile}_naive   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
    add_executable(${srcfile}_block   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
    set_target_properties(${srcfile}_block PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
    target_link_libraries(${srcfile}_naive ${exe_LIBS} gtest_main)
    target_link_libraries(${srcfile}_block ${exe_LIBS} gtest_main)

    gridtools_add_test( tests.${srcfile}_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_block" "12 33 61")
    gridtools_add_test( tests.${srcfile}_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_block" "23 11 43")
    gridtools_add_test( tests.${srcfile}_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_naive" "12 33 21")
    gridtools_add_test( tests.${srcfile}_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_naive" "23 11 23")
  endforeach(srcfile)

  add_executable(vanilla_laplacian  vanilla_laplacian.cpp)
  set_target_properties(vanilla_laplacian PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
  target_link_libraries(vanilla_laplacian ${exe_LIBS} )


  if ( ENABLE_CXX11 )
    foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS )
        string(TOLOWER ${variation} var_suffix)
        add_executable(interface1_functions_naive_${var_suffix}   interface1_functions.cpp ${GTEST_FILES})
        add_executable(interface1_functions_block_${var_suffix}   interface1_functions.cpp ${GTEST_FILES})
        set_target_properties(interface1_functions_naive_${var_suffix} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DFUNCTIONS_${variation}")
        set_target_properties(interface1_functions_block_${var_suffix} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DFUNCTIONS_${variation}")
        target_link_libraries(interface1_functions_naive_${var_suffix} ${exe_LIBS} gtest_main)
        target_link_libraries(interface1_functions_block_${var_suffix} ${exe_LIBS} gtest_main)
        gridtools_add_test( tests.interface1_functions_naive_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_naive_${var_suffix}" "12 33 61")
        gridtools_add_test( tests.interface1_functions_block_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_block_${var_suffix}" "23 11 43")
        gridtools_add_test( tests.interface1_functions_naive_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_naive_${var_suffix}" "12 33 21")
        gridtools_add_test( tests.interface1_functions_block_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_block_${var_suffix}" "23 11 23")
    endforeach(variation)

    add_executable(shallow_water_enhanced  shallow_water_enhanced.cpp)
    target_link_libraries(shallow_water_enhanced gcl ${ADD_MPI_LIBS} ${exe_LIBS} gtest gtest_main)
    #set_target_properties(shallow_water_enhanced PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
    if( NOT GCL_MPI ) # when MPI is on this test must be executed in parallel
      gridtools_add_test( tests.shallow_water_enhanced ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/shallow_water_enhanced" "8 8 1 1")
    endif(NOT GCL_MPI)

    if( GCL_MPI )
        set(ccflags " " )
        set(ldflags " ")

        #add_custom_mpi_host_test(shallow_water_enhanced_mpi ${CMAKE_CURRENT_SOURCE_DIR}/../examples/shallow_water_enhanced.cpp ${ldflags} ${ccflags})
        add_custom_mpi_host_test(copy_stencil_parallel ${CMAKE_CURRENT_SOURCE_DIR}/../examples/copy_stencil_parallel.cpp ${ldflags} ${ccflags})
    endif()
  endif()

  if(CUDA_FOUND)
    if(COMPILE_TO_PTX)
        CUDA_COMPILE_PTX(generated_files interface1.cu simple_hori_diff.cu copy_stencil.cu tridiagonal.cu boundary-condition.cu)
    else()

        foreach(srcfile IN LISTS SRCFILES )
             #set(CUDA_SEPARABLE_COMPILATION ON)
             cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu ${GTEST_FILES} OPTIONS ${GPU_SPECIFIC_FLAGS} )
             target_link_libraries(${srcfile}_cuda ${exe_LIBS} gtest_main)

             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_cuda" "77 48 61")
             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_cuda" "63 55 43")
        endforeach(srcfile)
     endif()

    if( ENABLE_CXX11 )
        foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS)
            string(TOLOWER ${variation} var_suffix)
            cuda_add_executable(interface1_functions_cuda_${var_suffix}   interface1_functions.cu ${GTEST_FILES} OPTIONS "-DFUNCTIONS_${variation}")
            target_link_libraries(interface1_functions_cuda_${var_suffix} ${exe_LIBS} gtest_main)
            gridtools_add_test( tests.interface1_functions_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_cuda_${var_suffix}" "23 11 43")
            gridtools_add_test( tests.interface1_functions_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_cuda_${var_suffix}" "23 11 23")
        endforeach(variation)

        cuda_add_executable(shallow_water_enhanced_cuda  shallow_water_enhanced.cu OPTIONS )
        target_link_libraries(shallow_water_enhanced_cuda ${exe_LIBS}  gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS} gtest gtest_main)
        gridtools_add_test( tests.shallow_water_enhanced_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/shallow_water_enhanced_cuda" "64 16 1 1")

        if( GCL_MPI )
          set(ccflags " " )
          set(ldflags " ")

          # TODO: not working
          # add_custom_mpi_gpu_test(shallow_water_enhanced_cuda_mpi ${CMAKE_CURRENT_SOURCE_DIR}/../examples/shallow_water_enhanced.cu ${ldflags} ${ccflags})
          add_custom_mpi_gpu_test(copy_stencil_parallel_cuda ${CMAKE_CURRENT_SOURCE_DIR}/../examples/copy_stencil_parallel.cu ${ldflags} ${ccflags})
          endif()
    endif()

    add_subdirectory( communication )
  endif()
endif(NOT STRUCTURED_GRIDS)
