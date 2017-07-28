if(NOT STRUCTURED_GRIDS)
    add_subdirectory(icosahedral)
else()

  file(GLOB_RECURSE INC_ALL_ "[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )
  file(GLOB_RECURSE INC_ALL_ "../include/[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )

  list( APPEND SRCFILES interface1 simple_hori_diff boundary-condition laplacian copy_stencil positional_copy_stencil tridiagonal vertical_advection_dycore alignment )
  list( APPEND X86_SRCFILES reduction)

  list(APPEND SRCFILES  extended_4D  copy_stencil_single_storage  expandable_parameters expandable_parameters_single_kernel expandable_parameters_reassign_domain data_store_field_test)

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

  if( GCL_MPI )
      set(ccflags " ${HOST_SPECIFIC_OPTIONS}" )
      set(ldflags " ")

      add_custom_mpi_host_test(copy_stencil_parallel ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cpp  ${ccflags} ${ldflags})

      add_custom_mpi_host_test(shallow_water_enhanced ${CMAKE_CURRENT_SOURCE_DIR}/./shallow_water_enhanced.cpp  ${ccflags} ${ldflags})

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

    foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS)
        string(TOLOWER ${variation} var_suffix)
        cuda_add_executable(interface1_functions_cuda_${var_suffix}   interface1_functions.cu ${GTEST_FILES} OPTIONS "-DFUNCTIONS_${variation} ${GPU_SPECIFIC_FLAGS}")
        target_link_libraries(interface1_functions_cuda_${var_suffix} ${exe_LIBS} gtest_main)
        gridtools_add_test( tests.interface1_functions_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_cuda_${var_suffix}" "23 11 43")
        gridtools_add_test( tests.interface1_functions_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_cuda_${var_suffix}" "23 11 23")
    endforeach(variation)
    if( GCL_MPI )
        set(ccflags " ")
        set(ldflags " ")
        add_custom_mpi_gpu_test(copy_stencil_parallel_cuda ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cu  ${ccflags} ${ldflags})

       add_custom_mpi_gpu_test(shallow_water_enhanced_cuda ${CMAKE_CURRENT_SOURCE_DIR}/./shallow_water_enhanced.cu  ${ccflags} ${ldflags})

     endif()

    add_subdirectory( communication )
  endif()
endif(NOT STRUCTURED_GRIDS)
