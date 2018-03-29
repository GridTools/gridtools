include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(NOT STRUCTURED_GRIDS)
    add_subdirectory(icosahedral)
else()
  file(GLOB_RECURSE INC_ALL_ "[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )
  file(GLOB_RECURSE INC_ALL_ "../include/[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )

  list( APPEND SRCFILES interface1 interface1_fused simple_hori_diff boundary-condition laplacian copy_stencil positional_copy_stencil tridiagonal vertical_advection_dycore alignment advection_pdbott_prepare_tracers )
  list( APPEND X86_SRCFILES reduction)

  list(APPEND SRCFILES  extended_4D  copy_stencil_single_storage  expandable_parameters expandable_parameters_single_kernel data_store_field_test)

  set( GTEST_FILES Options.hpp Options.cpp)

  add_definitions("-DBENCHMARK")

  if(ENABLE_HOST)
    foreach(srcfile IN LISTS SRCFILES X86_SRCFILES)
      add_executable(${srcfile}_host_naive   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
      add_executable(${srcfile}_host_block   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
      target_link_libraries(${srcfile}_host_naive ${exe_LIBS} gtest_main)
      target_link_libraries(${srcfile}_host_block ${exe_LIBS} gtest_main)
      target_compile_definitions(${srcfile}_host_naive PUBLIC ${HOST_BACKEND_DEFINE} "BACKEND_STRATEGY_NAIVE")
      target_compile_definitions(${srcfile}_host_block PUBLIC ${HOST_BACKEND_DEFINE})
  
      gridtools_add_test( tests.${srcfile}_host_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_host_block" "12 33 61")
      gridtools_add_test( tests.${srcfile}_host_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_host_block" "23 11 43")
      gridtools_add_test( tests.${srcfile}_host_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_host_naive" "12 33 21")
      gridtools_add_test( tests.${srcfile}_host_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_host_naive" "23 11 23")
    endforeach(srcfile)
  
    add_executable(vanilla_laplacian_host  vanilla_laplacian.cpp)
    set_target_properties(vanilla_laplacian_host PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
    target_link_libraries(vanilla_laplacian_host ${exe_LIBS} )
    target_compile_definitions(vanilla_laplacian_host PUBLIC ${HOST_BACKEND_DEFINE})
  
    foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS )
        string(TOLOWER ${variation} var_suffix)
        add_executable(interface1_functions_host_naive_${var_suffix}   interface1_functions.cpp ${GTEST_FILES})
        add_executable(interface1_functions_host_block_${var_suffix}   interface1_functions.cpp ${GTEST_FILES})
        set_target_properties(interface1_functions_host_naive_${var_suffix} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DFUNCTIONS_${variation}")
        set_target_properties(interface1_functions_host_block_${var_suffix} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DFUNCTIONS_${variation}")
        target_link_libraries(interface1_functions_host_naive_${var_suffix} ${exe_LIBS} gtest_main)
        target_link_libraries(interface1_functions_host_block_${var_suffix} ${exe_LIBS} gtest_main)
        target_compile_definitions(interface1_functions_host_naive_${var_suffix} PUBLIC ${HOST_BACKEND_DEFINE} "BACKEND_STRATEGY_NAIVE")
        target_compile_definitions(interface1_functions_host_block_${var_suffix} PUBLIC ${HOST_BACKEND_DEFINE})
        gridtools_add_test( tests.interface1_functions_host_naive_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_host_naive_${var_suffix}" "12 33 61")
        gridtools_add_test( tests.interface1_functions_host_block_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_host_block_${var_suffix}" "23 11 43")
        gridtools_add_test( tests.interface1_functions_host_naive_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_host_naive_${var_suffix}" "12 33 21")
        gridtools_add_test( tests.interface1_functions_host_block_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_host_block_${var_suffix}" "23 11 23")
    endforeach(variation)
  
    if( GCL_MPI )
        set(ccflags " ${HOST_SPECIFIC_OPTIONS}" )
        set(ldflags " ")
  
        add_custom_mpi_host_test(copy_stencil_parallel_host ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cpp  ${ccflags} ${ldflags})
  
        ## The next executable is not made into a test since it does not validate.
        ## The numerics need to be fixed (this is a task to be undertaken in the future)
        add_executable(shallow_water_enhanced_host  shallow_water_enhanced.cpp)
        target_link_libraries(shallow_water_enhanced_host ${exe_LIBS} gtest_main)
        set_target_properties(shallow_water_enhanced_host PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
        target_compile_definitions(shallow_water_enhanced_host PUBLIC ${HOST_BACKEND_DEFINE})
    endif()
  endif(ENABLE_HOST)

  if(ENABLE_MIC)
    foreach(srcfile IN LISTS SRCFILES X86_SRCFILES)
      add_executable(${srcfile}_mic   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
      target_link_libraries(${srcfile}_mic ${exe_LIBS} gtest_main)
      target_compile_definitions(${srcfile}_mic PUBLIC ${MIC_BACKEND_DEFINE})

      gridtools_add_test( tests.${srcfile}_mic ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_mic" "12 33 61")
      gridtools_add_test( tests.${srcfile}_mic ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_mic" "23 11 43")
    endforeach(srcfile)

    add_executable(vanilla_laplacian_mic  vanilla_laplacian.cpp)
    set_target_properties(vanilla_laplacian_mic PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
    target_link_libraries(vanilla_laplacian_mic ${exe_LIBS} )
    target_compile_definitions(vanilla_laplacian_mic PUBLIC ${MIC_BACKEND_DEFINE})

    foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS )
        string(TOLOWER ${variation} var_suffix)
        add_executable(interface1_functions_mic_${var_suffix}   interface1_functions.cpp ${GTEST_FILES})
        set_target_properties(interface1_functions_mic_${var_suffix} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DFUNCTIONS_${variation}")
        target_link_libraries(interface1_functions_mic_${var_suffix} ${exe_LIBS} gtest_main)
        target_compile_definitions(interface1_functions_mic_${var_suffix} PUBLIC ${MIC_BACKEND_DEFINE})
        gridtools_add_test( tests.interface1_functions_mic_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_mic_${var_suffix}" "23 11 43")
        gridtools_add_test( tests.interface1_functions_mic_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_mic_${var_suffix}" "23 11 23")
    endforeach(variation)

    if( GCL_MPI )
        set(ccflags " ${MIC_SPECIFIC_OPTIONS}" )
        set(ldflags " ")

        add_custom_mpi_mic_test(copy_stencil_parallel_mic ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cpp  ${ccflags} ${ldflags})

        ## The next executable is not made into a test since it does not validate.
        ## The numerics need to be fixed (this is a task to be undertaken in the future)
        add_executable(shallow_water_enhanced_mic  shallow_water_enhanced.cpp)
        target_link_libraries(shallow_water_enhanced_mic ${exe_LIBS} gtest_main)
        set_target_properties(shallow_water_enhanced_mic PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
        target_compile_definitions(shallow_water_enhanced_mic PUBLIC ${MIC_BACKEND_DEFINE})
    endif()
  endif(ENABLE_MIC)

  add_subdirectory( c_bindings )

  if(ENABLE_CUDA)

      if(COMPILE_TO_PTX)
        CUDA_COMPILE_PTX(generated_files interface1.cu simple_hori_diff.cu copy_stencil.cu tridiagonal.cu boundary-condition.cu)
    else()

        foreach(srcfile IN LISTS SRCFILES )
             #set(CUDA_SEPARABLE_COMPILATION ON)
             cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu ${GTEST_FILES} OPTIONS ${GPU_SPECIFIC_FLAGS} "-D${CUDA_BACKEND_DEFINE}")
             target_link_libraries(${srcfile}_cuda ${exe_LIBS} gtest_main)

             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_cuda" "47 38 61")
             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${srcfile}_cuda" "63 38 10")
        endforeach(srcfile)
     endif()

    foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS)
        string(TOLOWER ${variation} var_suffix)
        cuda_add_executable(interface1_functions_cuda_${var_suffix}   interface1_functions.cu ${GTEST_FILES} OPTIONS "-DFUNCTIONS_${variation} ${GPU_SPECIFIC_FLAGS}" "-D${CUDA_BACKEND_DEFINE}")
        target_link_libraries(interface1_functions_cuda_${var_suffix} ${exe_LIBS} gtest_main)
        gridtools_add_test( tests.interface1_functions_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_cuda_${var_suffix}" "23 38 43")
        gridtools_add_test( tests.interface1_functions_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface1_functions_cuda_${var_suffix}" "23 38 23")
    endforeach(variation)
    if( GCL_MPI )
        set(ccflags " ")
        set(ldflags " ")
        add_custom_mpi_gpu_test(copy_stencil_parallel_cuda ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cu  ${ccflags} ${ldflags})

        ## The next executable is not made into a test since it does not validate.
        ## The numerics need to be fixed (this is a task to be undertaken in the future)
        cuda_add_executable(shallow_water_enhanced_cuda  shallow_water_enhanced.cu OPTIONS "${GPU_SPECIFIC_FLAGS} -D_GCL_MPI_" "-D${CUDA_BACKEND_DEFINE}")
        target_link_libraries(shallow_water_enhanced_cuda ${exe_LIBS} gtest_main)
     endif()

    add_subdirectory( communication )
endif(ENABLE_CUDA)
endif(NOT STRUCTURED_GRIDS)
