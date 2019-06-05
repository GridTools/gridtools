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
          layout_transformation
          boundary_conditions
          )
      set(SOURCES
          ${SOURCES_PERFTEST}
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

      if(GT_ENABLE_BACKEND_X86)
        foreach(srcfile IN LISTS SOURCES)
          add_executable(${srcfile}_x86 ${srcfile}.cpp)
          target_link_libraries(${srcfile}_x86 regression_main GridToolsTestX86)

          gridtools_add_test(
              NAME tests.${srcfile}_x86_12_33_61
              COMMAND $<TARGET_FILE:${srcfile}_x86> 12 33 61
              LABELS regression_x86 backend_x86
              )
          gridtools_add_test(
              NAME tests.${srcfile}_x86_23_11_43
              COMMAND $<TARGET_FILE:${srcfile}_x86> 23 11 43
              LABELS regression_x86 backend_x86
              )

          if (srcfile IN_LIST SOURCES_PERFTEST)
              add_dependencies(perftests ${srcfile}_x86)
              set_property(TEST tests.${srcfile}_x86_23_11_43 APPEND PROPERTY LABELS perftests_x86)
          endif()
        endforeach(srcfile)

        if( GT_USE_MPI )
            add_custom_mpi_test(x86 TARGET copy_stencil_parallel NPROC 4 SOURCES copy_stencil_parallel.cpp)

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_x86  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_x86 gtest_main gcl GridToolsTestX86)
        endif()
      endif(GT_ENABLE_BACKEND_X86)

      if(GT_ENABLE_BACKEND_NAIVE)
        foreach(srcfile IN LISTS SOURCES)
          add_executable(${srcfile}_naive   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_naive regression_main GridToolsTestNAIVE)

          gridtools_add_test(
              NAME tests.${srcfile}_naive_12_33_61
              COMMAND $<TARGET_FILE:${srcfile}_naive> 12 33 21
              LABELS regression_naive backend_naive
              )
          gridtools_add_test(
              NAME tests.${srcfile}_naive_23_11_43
              COMMAND $<TARGET_FILE:${srcfile}_naive> 23 11 43
              LABELS regression_naive backend_naive
              )
        endforeach(srcfile)

        if( GT_USE_MPI )
            add_custom_mpi_test(naive TARGET copy_stencil_parallel NPROC 4 SOURCES copy_stencil_parallel.cpp)

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_naive  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_naive gtest_main gcl GridToolsTestNAIVE)
        endif()
      endif(GT_ENABLE_BACKEND_NAIVE)

      if(GT_ENABLE_BACKEND_MC)
        foreach(srcfile IN LISTS SOURCES)
          add_executable(${srcfile}_mc   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_mc regression_main GridToolsTestMC)

          gridtools_add_test(
              NAME tests.${srcfile}_mc_12_33_61
              COMMAND $<TARGET_FILE:${srcfile}_mc> 12 33 61
              LABELS regression_mc backend_mc
              )
          gridtools_add_test(
              NAME tests.${srcfile}_mc_23_11_43
              COMMAND $<TARGET_FILE:${srcfile}_mc> 23 11 43
              LABELS regression_mc backend_mc
              )
          if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_mc)
            set_property(TEST tests.${srcfile}_mc_23_11_43 APPEND PROPERTY LABELS perftests_mc)
          endif()
        endforeach(srcfile)

        if( GT_USE_MPI )
            add_custom_mpi_test(mc TARGET copy_stencil_parallel NPROC 4 SOURCES copy_stencil_parallel.cpp)

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_mc  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_mc  gtest_main gcl GridToolsTestMC)
        endif()
    endif(GT_ENABLE_BACKEND_MC)

    if(GT_ENABLE_BACKEND_CUDA)

       foreach(srcfile IN LISTS SOURCES)
           add_executable( ${srcfile}_cuda ${srcfile}.cu)
           target_link_libraries(${srcfile}_cuda regression_main GridToolsTestCUDA)

           gridtools_add_test(
               NAME tests.${srcfile}_cuda_47_38_61
               COMMAND $<TARGET_FILE:${srcfile}_cuda> 47 38 61
               LABELS regression_cuda backend_cuda
               )
           gridtools_add_test(
               NAME tests.${srcfile}_cuda_63_38_10
               COMMAND $<TARGET_FILE:${srcfile}_cuda> 63 38 10
               LABELS regression_cuda backend_cuda
               )
           if (srcfile IN_LIST SOURCES_PERFTEST)
             add_dependencies(perftests ${srcfile}_cuda)
             set_property(TEST tests.${srcfile}_cuda_63_38_10 APPEND PROPERTY LABELS perftests_cuda)
           endif()

           if (GT_CUDA_PTX_GENERATION)
               add_library( ${srcfile}_cuda_ptx OBJECT ${srcfile}.cu)
               target_link_libraries(${srcfile}_cuda_ptx regression_main GridToolsTestCUDA)
               set_target_properties(${srcfile}_cuda_ptx PROPERTIES CUDA_PTX_COMPILATION ON)
           endif()
       endforeach(srcfile)

       if( GT_USE_MPI )
           add_custom_mpi_test(cuda TARGET copy_stencil_parallel NPROC 4 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/copy_stencil_parallel.cu )

           ## The next executable is not made into a test since it does not validate.
           ## The numerics need to be fixed (this is a task to be undertaken in the future)
           add_executable(shallow_water_enhanced_cuda shallow_water_enhanced.cu)
           target_link_libraries(shallow_water_enhanced_cuda gtest_main gcl GridToolsTestCUDA)
        endif()
    endif(GT_ENABLE_BACKEND_CUDA)

    if ( COMPONENT_C_BINDINGS )
        add_subdirectory( c_bindings )
    endif()

    if ( COMPONENT_GCL )
        add_subdirectory( communication )
    endif()

  endif(GT_TESTS_ICOSAHEDRAL_GRID)
endif( COMPONENT_STENCIL_COMPOSITION )
