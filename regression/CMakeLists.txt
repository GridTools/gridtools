include_directories(${CMAKE_SOURCE_DIR}/include)

if( COMPONENT_STENCIL_COMPOSITION )
if(NOT STRUCTURED_GRIDS)
    add_subdirectory(icosahedral)
else()
  file(GLOB_RECURSE INC_ALL_ "[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )
  file(GLOB_RECURSE INC_ALL_ "../include/[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )

  list(APPEND PERFTEST_SRCFILES interface1 interface1_fused simple_hori_diff copy_stencil vertical_advection_dycore advection_pdbott_prepare_tracers)
  list(APPEND SRCFILES boundary-condition laplacian positional_copy_stencil tridiagonal alignment)
  list(APPEND SRCFILES extended_4D expandable_parameters expandable_parameters_single_kernel)
  list(APPEND X86_SRCFILES reduction)

  set( GTEST_FILES Options.hpp Options.cpp)

  add_definitions("-DBENCHMARK")

  # special target for executables which are used from performance benchmarks
  add_custom_target(perftests)

  if(ENABLE_HOST)
    foreach(srcfile IN LISTS PERFTEST_SRCFILES SRCFILES X86_SRCFILES)
      add_executable(${srcfile}_host_naive   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
      add_executable(${srcfile}_host_block   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
      target_link_libraries(${srcfile}_host_naive ${exe_LIBS} gtest_main)
      target_link_libraries(${srcfile}_host_block ${exe_LIBS} gtest_main)
      set(naive_flags ${GT_CXX_FLAGS};-D${HOST_BACKEND_DEFINE};-DBACKEND_STRATEGY_NAIVE)
      target_compile_options(${srcfile}_host_naive PRIVATE ${GT_CXX_FLAGS};-D${HOST_BACKEND_DEFINE};-DBACKEND_STRATEGY_NAIVE)
      target_compile_options(${srcfile}_host_block PRIVATE ${GT_CXX_FLAGS} -D${HOST_BACKEND_DEFINE})

      gridtools_add_test( tests.${srcfile}_host_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_host_block" "12 33 61")
      gridtools_add_test( tests.${srcfile}_host_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_host_block" "23 11 43")
      gridtools_add_test( tests.${srcfile}_host_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_host_naive" "12 33 21")
      gridtools_add_test( tests.${srcfile}_host_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_host_naive" "23 11 23")
    endforeach(srcfile)
    foreach(perftest IN LISTS PERFTEST_SRCFILES)
      add_dependencies(perftests ${perftest}_host_block)
    endforeach()

    add_executable(vanilla_laplacian_host  vanilla_laplacian.cpp)
    target_compile_options(vanilla_laplacian_host PRIVATE ${GT_CXX_FLAGS})
    target_link_libraries(vanilla_laplacian_host ${exe_LIBS} )
    target_compile_definitions(vanilla_laplacian_host PUBLIC ${HOST_BACKEND_DEFINE})

    foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS )
        string(TOLOWER ${variation} var_suffix)
        add_executable(interface1_functions_host_naive_${var_suffix}   interface1_functions.cpp ${GTEST_FILES})
        add_executable(interface1_functions_host_block_${var_suffix}   interface1_functions.cpp ${GTEST_FILES})
        target_compile_options(interface1_functions_host_naive_${var_suffix} PRIVATE ${GT_CXX_FLAGS};-D${HOST_BACKEND_DEFINE};-DBACKEND_STRATEGY_NAIVE;-DFUNCTIONS_${variation})
        target_compile_options(interface1_functions_host_block_${var_suffix} PRIVATE ${GT_CXX_FLAGS};-D${HOST_BACKEND_DEFINE};-DFUNCTIONS_${variation})
        target_link_libraries(interface1_functions_host_naive_${var_suffix} ${exe_LIBS} gtest_main)
        target_link_libraries(interface1_functions_host_block_${var_suffix} ${exe_LIBS} gtest_main)

        gridtools_add_test( tests.interface1_functions_host_naive_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/interface1_functions_host_naive_${var_suffix}" "12 33 61")
        gridtools_add_test( tests.interface1_functions_host_block_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/interface1_functions_host_block_${var_suffix}" "23 11 43")
        gridtools_add_test( tests.interface1_functions_host_naive_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/interface1_functions_host_naive_${var_suffix}" "12 33 21")
        gridtools_add_test( tests.interface1_functions_host_block_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/interface1_functions_host_block_${var_suffix}" "23 11 23")
    endforeach(variation)

    if( GCL_MPI )
        set(ccflags " ${HOST_SPECIFIC_OPTIONS}" )
        set(ldflags " ")

        add_custom_mpi_host_test(copy_stencil_parallel_host ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cpp  ${ccflags} ${ldflags})

        ## The next executable is not made into a test since it does not validate.
        ## The numerics need to be fixed (this is a task to be undertaken in the future)
        add_executable(shallow_water_enhanced_host  shallow_water_enhanced.cpp)
        target_link_libraries(shallow_water_enhanced_host ${exe_LIBS} gtest_main)
        target_compile_options(shallow_water_enhanced_host PRIVATE ${GT_CXX_FLAGS};-D${HOST_BACKEND_DEFINE})    
    endif()
  endif(ENABLE_HOST)

#####   if(ENABLE_MIC)
#####     foreach(srcfile IN LISTS PERFTEST_SRCFILES SRCFILES X86_SRCFILES)
#####       add_executable(${srcfile}_mic   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
#####       target_link_libraries(${srcfile}_mic ${exe_LIBS} gtest_main)
#####       target_compile_definitions(${srcfile}_mic PUBLIC ${GT_CXX_FLAGS} ${MIC_BACKEND_DEFINE})

#####       gridtools_add_test( tests.${srcfile}_mic ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_mic" "12 33 61")
#####       gridtools_add_test( tests.${srcfile}_mic ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_mic" "23 11 43")
#####     endforeach(srcfile)
#####     foreach(perftest IN LISTS PERFTEST_SRCFILES)
#####       add_dependencies(perftests ${perftest}_mic)
#####     endforeach()


#####     add_executable(vanilla_laplacian_mic  vanilla_laplacian.cpp)
#####     set_target_properties(vanilla_laplacian_mic PROPERTIES COMPILE_FLAGS ${GT_CXX_FLAGS})
#####     target_link_libraries(vanilla_laplacian_mic ${exe_LIBS} )
#####     target_compile_definitions(vanilla_laplacian_mic PUBLIC ${MIC_BACKEND_DEFINE})

#####     foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS )
#####         string(TOLOWER ${variation} var_suffix)
#####         add_executable(interface1_functions_mic_${var_suffix}   interface1_functions.cpp ${GTEST_FILES})
#####         set_target_properties(interface1_functions_mic_${var_suffix} PROPERTIES COMPILE_FLAGS "${GT_CXX_FLAGS} -DFUNCTIONS_${variation}")
#####         target_link_libraries(interface1_functions_mic_${var_suffix} ${exe_LIBS} gtest_main)
#####         set_target_properties(interface1_functions_mic_${var_suffix} PROPERTIES COMPILE_FLAGS "${GT_CXX_FLAGS} -D${MIC_BACKEND_DEFINE}")
#####         gridtools_add_test( tests.interface1_functions_mic_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/interface1_functions_mic_${var_suffix}" "23 11 43")
#####     endforeach(variation)

#####     if( GCL_MPI )
#####         set(ccflags " ${MIC_SPECIFIC_OPTIONS}" )
#####         set(ldflags " ")

#####         add_custom_mpi_mic_test(copy_stencil_parallel_mic ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cpp  ${ccflags} ${ldflags})

#####         ## The next executable is not made into a test since it does not validate.
#####         ## The numerics need to be fixed (this is a task to be undertaken in the future)
#####         add_executable(shallow_water_enhanced_mic  shallow_water_enhanced.cpp)
#####         target_link_libraries(shallow_water_enhanced_mic ${exe_LIBS} gtest_main)
#####         set_target_properties(shallow_water_enhanced_mic PROPERTIES COMPILE_FLAGS "${GT_CXX_FLAGS} -D${MIC_BACKEND_DEFINE}")
#####     endif()
#####   endif(ENABLE_MIC)

#####   if(ENABLE_CUDA)

#####       if(COMPILE_TO_PTX)
#####         CUDA_COMPILE_PTX(generated_files interface1.cu simple_hori_diff.cu copy_stencil.cu tridiagonal.cu boundary-condition.cu)
#####       else()
#####         foreach(srcfile IN LISTS PERFTEST_SRCFILES SRCFILES )
#####              #set(CUDA_SEPARABLE_COMPILATION ON)
#####              cuda_add_executable( ${srcfile}_cuda ${srcfile}.cu ${GTEST_FILES} OPTIONS ${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS} "-D${CUDA_BACKEND_DEFINE}")
#####              target_link_libraries(${srcfile}_cuda ${exe_LIBS} gtest_main)

#####              gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_cuda" "47 38 61")
#####              gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_cuda" "63 38 10")
#####         endforeach(srcfile)
#####         foreach(perftest IN LISTS PERFTEST_SRCFILES)
#####           add_dependencies(perftests ${perftest}_cuda)
#####         endforeach()
#####      endif()

#####     foreach( variation CALL MONOLITHIC OFFSETS PROCEDURES PROCEDURES_OFFSETS)
#####         string(TOLOWER ${variation} var_suffix)
#####         cuda_add_executable(interface1_functions_cuda_${var_suffix}   interface1_functions.cu ${GTEST_FILES} OPTIONS "${GT_CXX_FLAGS} -DFUNCTIONS_${variation} ${GPU_SPECIFIC_FLAGS}" "-D${CUDA_BACKEND_DEFINE}")
#####         target_link_libraries(interface1_functions_cuda_${var_suffix} ${exe_LIBS} gtest_main)
#####         gridtools_add_test( tests.interface1_functions_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/interface1_functions_cuda_${var_suffix}" "23 38 43")
#####         gridtools_add_test( tests.interface1_functions_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/interface1_functions_cuda_${var_suffix}" "23 38 23")
#####     endforeach(variation)
#####     if( GCL_MPI )
#####         set(ccflags " ")
#####         set(ldflags " ")
#####         add_custom_mpi_gpu_test(copy_stencil_parallel_cuda ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cu  ${ccflags} ${ldflags})

#####         ## The next executable is not made into a test since it does not validate.
#####         ## The numerics need to be fixed (this is a task to be undertaken in the future)
#####         cuda_add_executable(shallow_water_enhanced_cuda  shallow_water_enhanced.cu OPTIONS "${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS} -D_GCL_MPI_" "-D${CUDA_BACKEND_DEFINE}")
#####         target_link_libraries(shallow_water_enhanced_cuda ${exe_LIBS} gtest_main)
#####      endif()
##### endif(ENABLE_CUDA)

endif(NOT STRUCTURED_GRIDS)
endif( COMPONENT_STENCIL_COMPOSITION )

if ( COMPONENT_GCL )
    add_subdirectory( communication )
endif()

if ( COMPONENT_C_BINDINGS )
  add_subdirectory( c_bindings )
endif()
