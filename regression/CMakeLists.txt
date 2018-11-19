if( COMPONENT_STENCIL_COMPOSITION )

    if(NOT GT_TESTS_STRUCTURED_GRIDS)
        add_subdirectory(icosahedral)
    else()

      set(SOURCES_PERFTEST
          horizontal_diffusion
          horizontal_diffusion_fused
          simple_hori_diff
          copy_stencil
          vertical_advection_dycore
          advection_pdbott_prepare_tracers
          )
      set(SOURCES_CUDA
          ${SOURCES_PERFTEST}
          boundary-condition
          laplacian positional_stencil
          tridiagonal
          alignment
          extended_4D
          expandable_parameters
          expandable_parameters_single_kernel
          horizontal_diffusion_functions
          )
      set(SOURCES_X86
          ${SOURCES_CUDA}
          reduction
          )

      # special target for executables which are used from performance benchmarks
      add_custom_target(perftests)

      if(GT_ENABLE_TARGET_X86)
        foreach(srcfile IN LISTS SOURCES_X86)
          add_executable(${srcfile}_x86_naive   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_x86_naive regression_fixture GridToolsTestX86)
          target_compile_definitions(${srcfile}_x86_naive PRIVATE BACKEND_STRATEGY_NAIVE)

          add_executable(${srcfile}_x86_block   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_x86_block regression_fixture GridToolsTestX86)

          gridtools_add_test(
              NAME tests.${srcfile}_x86_block_12_33_61
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_x86_block> 12 33 61
              LABELS regression_x86 target_x86
              )
          gridtools_add_test(
              NAME tests.${srcfile}_x86_block_23_11_43
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_x86_block> 23 11 43
              LABELS regression_x86 target_x86
              )
          gridtools_add_test(
              NAME tests.${srcfile}_x86_naive_12_33_61
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_x86_naive> 12 33 21
              LABELS regression_x86 target_x86
              )
          gridtools_add_test(
              NAME tests.${srcfile}_x86_naive_23_11_43
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_x86_naive> 23 11 43
              LABELS regression_x86 target_x86
              )

          if (srcfile IN_LIST SOURCES_PERFTEST)
              add_dependencies(perftests ${srcfile}_x86_block)
          endif()
        endforeach(srcfile)

        add_executable(vanilla_laplacian_x86  vanilla_laplacian.cpp)
        target_link_libraries(vanilla_laplacian_x86 GridToolsTestX86)

        if(GT_USE_MPI)
            add_custom_mpi_x86_test(TARGET copy_stencil_parallel SOURCES copy_stencil_parallel.cpp)

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_x86  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_x86 gtest_main gcl GridToolsTestX86)
        endif()
      endif(GT_ENABLE_TARGET_X86)

      if(GT_ENABLE_TARGET_MC)
        foreach(srcfile IN LISTS SOURCES_X86)
          add_executable(${srcfile}_mc   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_mc regression_fixture GridToolsTestMC)

          gridtools_add_test(
              NAME tests.${srcfile}_mc_12_33_61
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_mc> 12 33 61
              LABELS regression_mc target_mc
              )
          gridtools_add_test(
              NAME tests.${srcfile}_mc_23_11_43
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_mc> 23 11 43
              LABELS regression_mc target_mc
              )
          if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_mc)
          endif()
        endforeach(srcfile)


        add_executable(vanilla_laplacian_mc  vanilla_laplacian.cpp)
        target_link_libraries(vanilla_laplacian_mc GridToolsTestMC)

        if(GT_USE_MPI)
            add_custom_mpi_mc_test(TARGET copy_stencil_parallel SOURCES copy_stencil_parallel.cpp)

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_mc  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_mc  gtest_main gcl GridToolsTestMC)
        endif()
    endif(GT_ENABLE_TARGET_MC)

    if(GT_ENABLE_TARGET_CUDA)

       foreach(srcfile IN LISTS SOURCES_CUDA)
           add_executable( ${srcfile}_cuda ${srcfile}.cu)
           target_link_libraries(${srcfile}_cuda regression_fixture GridToolsTestCUDA)

           gridtools_add_test(
               NAME tests.${srcfile}_cuda_47_38_61
               SCRIPT ${TEST_SCRIPT}
               COMMAND $<TARGET_FILE:${srcfile}_cuda> 47 38 61
               LABELS regression_cuda target_cuda
               )
           gridtools_add_test(
               NAME tests.${srcfile}_cuda_63_38_10
               SCRIPT ${TEST_SCRIPT}
               COMMAND $<TARGET_FILE:${srcfile}_cuda> 63 38 10
               LABELS regression_cuda target_cuda
               )
           if (srcfile IN_LIST SOURCES_PERFTEST)
             add_dependencies(perftests ${srcfile}_cuda)
           endif()

           if (GT_CUDA_PTX_GENERATION)
               add_library( ${srcfile}_cuda_ptx OBJECT ${srcfile}.cu)
               target_link_libraries(${srcfile}_cuda_ptx regression_fixture GridToolsTestCUDA)
               set_target_properties(${srcfile}_cuda_ptx PROPERTIES CUDA_PTX_COMPILATION ON)
           endif()
       endforeach(srcfile)

       if(GT_USE_MPI)
           add_custom_mpi_gpu_test(TARGET copy_stencil_parallel SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/copy_stencil_parallel.cu )

           ## The next executable is not made into a test since it does not validate.
           ## The numerics need to be fixed (this is a task to be undertaken in the future)
           add_executable(shallow_water_enhanced_cuda shallow_water_enhanced.cu)
           target_link_libraries(shallow_water_enhanced_cuda gtest_main gcl GridToolsTestCUDA)
        endif()
    endif(GT_ENABLE_TARGET_CUDA)

    if ( COMPONENT_C_BINDINGS )
        add_subdirectory( c_bindings )
    endif()

    if ( COMPONENT_GCL )
        add_subdirectory( communication )
    endif()

  endif(NOT GT_TESTS_STRUCTURED_GRIDS)
endif( COMPONENT_STENCIL_COMPOSITION )
