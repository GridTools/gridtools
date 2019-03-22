if( COMPONENT_STENCIL_COMPOSITION )

    if(GT_TESTS_ICOSAHEDRAL_GRID)
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
      set(SOURCES
          ${SOURCES_PERFTEST}
          boundary_condition
          laplacian positional_stencil
          tridiagonal
          alignment
          extended_4D
          expandable_parameters
          expandable_parameters_single_kernel
          horizontal_diffusion_functions
          )

      # special target for executables which are used from performance benchmarks
      add_custom_target(perftests)

      if(GT_ENABLE_TARGET_X86)
        foreach(srcfile IN LISTS SOURCES)
          add_executable(${srcfile}_x86 ${srcfile}.cpp)
          target_link_libraries(${srcfile}_x86 regression_main GridToolsTestX86)

          gridtools_add_test(
              NAME tests.${srcfile}_x86_12_33_61
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_x86> 12 33 61
              LABELS regression_x86 target_x86
              ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
              )
          gridtools_add_test(
              NAME tests.${srcfile}_x86_23_11_43
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_x86> 23 11 43
              LABELS regression_x86 target_x86
              ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
              )

          if (srcfile IN_LIST SOURCES_PERFTEST)
              add_dependencies(perftests ${srcfile}_x86)
          endif()
        endforeach(srcfile)

        if( GT_USE_MPI )
            add_custom_mpi_x86_test(TARGET copy_stencil_parallel NPROC 4 SOURCES copy_stencil_parallel.cpp)

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_x86  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_x86 gtest_main gcl GridToolsTestX86)
        endif()
      endif(GT_ENABLE_TARGET_X86)

      if(GT_ENABLE_TARGET_NAIVE)
        foreach(srcfile IN LISTS SOURCES)
          add_executable(${srcfile}_naive   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_naive regression_main GridToolsTestNaive)

          gridtools_add_test(
              NAME tests.${srcfile}_naive_12_33_61
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_naive> 12 33 21
              LABELS regression_naive target_naive
              ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
              )
          gridtools_add_test(
              NAME tests.${srcfile}_naive_23_11_43
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_naive> 23 11 43
              LABELS regression_naive target_naive
              ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
              )
        endforeach(srcfile)

        if( GT_USE_MPI )
            add_custom_mpi_naive_test(TARGET copy_stencil_parallel NPROC 4 SOURCES copy_stencil_parallel.cpp)

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_naive  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_naive gtest_main gcl GridToolsTestNaive)
        endif()
      endif(GT_ENABLE_TARGET_NAIVE)

      if(GT_ENABLE_TARGET_MC)
        foreach(srcfile IN LISTS SOURCES)
          add_executable(${srcfile}_mc   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_mc regression_main GridToolsTestMC)

          gridtools_add_test(
              NAME tests.${srcfile}_mc_12_33_61
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_mc> 12 33 61
              LABELS regression_mc target_mc
              ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
              )
          gridtools_add_test(
              NAME tests.${srcfile}_mc_23_11_43
              SCRIPT ${TEST_SCRIPT}
              COMMAND $<TARGET_FILE:${srcfile}_mc> 23 11 43
              LABELS regression_mc target_mc
              ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
              )
          if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_mc)
          endif()
        endforeach(srcfile)

        if( GT_USE_MPI )
            add_custom_mpi_mc_test(TARGET copy_stencil_parallel NPROC 4 SOURCES copy_stencil_parallel.cpp)

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_mc  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_mc  gtest_main gcl GridToolsTestMC)
        endif()
    endif(GT_ENABLE_TARGET_MC)

    if(GT_ENABLE_TARGET_CUDA)

       foreach(srcfile IN LISTS SOURCES)
           add_executable( ${srcfile}_cuda ${srcfile}.cu)
           target_link_libraries(${srcfile}_cuda regression_main GridToolsTestCUDA)

           gridtools_add_test(
               NAME tests.${srcfile}_cuda_47_38_61
               SCRIPT ${TEST_SCRIPT}
               COMMAND $<TARGET_FILE:${srcfile}_cuda> 47 38 61
               LABELS regression_cuda target_cuda
               ENVIRONMENT ${TEST_CUDA_ENVIRONMENT}
               )
           gridtools_add_test(
               NAME tests.${srcfile}_cuda_63_38_10
               SCRIPT ${TEST_SCRIPT}
               COMMAND $<TARGET_FILE:${srcfile}_cuda> 63 38 10
               LABELS regression_cuda target_cuda
               ENVIRONMENT ${TEST_CUDA_ENVIRONMENT}
               )
           if (srcfile IN_LIST SOURCES_PERFTEST)
             add_dependencies(perftests ${srcfile}_cuda)
           endif()

           if (GT_CUDA_PTX_GENERATION)
               add_library( ${srcfile}_cuda_ptx OBJECT ${srcfile}.cu)
               target_link_libraries(${srcfile}_cuda_ptx regression_main GridToolsTestCUDA)
               set_target_properties(${srcfile}_cuda_ptx PROPERTIES CUDA_PTX_COMPILATION ON)
           endif()
       endforeach(srcfile)

       if( GT_USE_MPI )
           add_custom_mpi_gpu_test(TARGET copy_stencil_parallel NPROC 4 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/copy_stencil_parallel.cu )

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

  endif(GT_TESTS_ICOSAHEDRAL_GRID)
endif( COMPONENT_STENCIL_COMPOSITION )
