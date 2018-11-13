include_directories(${CMAKE_SOURCE_DIR}/include)

if( COMPONENT_STENCIL_COMPOSITION )

    if(NOT STRUCTURED_GRIDS)
        add_subdirectory(icosahedral)
    else()

      list(APPEND PERFTEST_SRCFILES horizontal_diffusion horizontal_diffusion_fused simple_hori_diff copy_stencil vertical_advection_dycore advection_pdbott_prepare_tracers)
      list(APPEND SRCFILES boundary-condition laplacian positional_stencil tridiagonal alignment)
      list(APPEND SRCFILES extended_4D expandable_parameters expandable_parameters_single_kernel horizontal_diffusion_functions)
      list(APPEND X86_SRCFILES reduction)

      add_definitions("-DBENCHMARK")

      # special target for executables which are used from performance benchmarks
      add_custom_target(perftests)

      if(GT_ENABLE_TARGET_X86)
        foreach(srcfile IN LISTS PERFTEST_SRCFILES SRCFILES X86_SRCFILES)
          add_executable(${srcfile}_x86_naive   ${srcfile}.cpp)
          add_executable(${srcfile}_x86_block   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_x86_naive regression_fixture stencil-composition)
          target_link_libraries(${srcfile}_x86_block regression_fixture stencil-composition)
          target_compile_options(${srcfile}_x86_naive PRIVATE ${GT_CXX_FLAGS} ${GT_CXX_HOST_ONLY_FLAGS} -D${X86_BACKEND_DEFINE} -DBACKEND_STRATEGY_NAIVE)
          target_compile_options(${srcfile}_x86_block PRIVATE ${GT_CXX_FLAGS} ${GT_CXX_HOST_ONLY_FLAGS} -D${X86_BACKEND_DEFINE})

          gridtools_add_test( tests.${srcfile}_x86_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_x86_block" "12 33 61")
          gridtools_add_test( tests.${srcfile}_x86_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_x86_block" "23 11 43")
          gridtools_add_test( tests.${srcfile}_x86_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_x86_naive" "12 33 21")
          gridtools_add_test( tests.${srcfile}_x86_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_x86_naive" "23 11 23")
        endforeach(srcfile)
        foreach(perftest IN LISTS PERFTEST_SRCFILES)
          add_dependencies(perftests ${perftest}_x86_block)
        endforeach()

        add_executable(vanilla_laplacian_x86  vanilla_laplacian.cpp)
        target_compile_options(vanilla_laplacian_x86 PRIVATE ${GT_CXX_FLAGS})
        target_link_libraries(vanilla_laplacian_x86 stencil-composition)
        target_compile_definitions(vanilla_laplacian_x86 PUBLIC ${X86_BACKEND_DEFINE})

        if( GCL_MPI )
            set(ccflags ${X86_SPECIFIC_OPTIONS} )

            add_custom_mpi_x86_test(TARGET copy_stencil_parallel SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cpp  ADDITIONAL_FLAGS ${ccflags})

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_x86  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_x86 gtest_main gcl)
            target_compile_options(shallow_water_enhanced_x86 PRIVATE ${GT_CXX_FLAGS} ${GT_CXX_HOST_ONLY_FLAGS} -D${X86_BACKEND_DEFINE})
        endif()
      endif(GT_ENABLE_TARGET_X86)

      if(GT_ENABLE_TARGET_MC)
        foreach(srcfile IN LISTS PERFTEST_SRCFILES SRCFILES X86_SRCFILES)
          add_executable(${srcfile}_mc   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_mc regression_fixture stencil-composition)
          target_compile_options(${srcfile}_mc PUBLIC ${GT_CXX_FLAGS} ${GT_CXX_HOST_ONLY_FLAGS} -D${MC_BACKEND_DEFINE})

          gridtools_add_test( tests.${srcfile}_mc ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_mc" "12 33 61")
          gridtools_add_test( tests.${srcfile}_mc ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_mc" "23 11 43")
        endforeach(srcfile)
        foreach(perftest IN LISTS PERFTEST_SRCFILES)
          add_dependencies(perftests ${perftest}_mc)
        endforeach()


        add_executable(vanilla_laplacian_mc  vanilla_laplacian.cpp)
        target_link_libraries(vanilla_laplacian_mc stencil-composition)
        target_compile_options(vanilla_laplacian_mc PUBLIC ${GT_CXX_FLAGS})
        target_compile_definitions(vanilla_laplacian_mc PUBLIC ${MC_BACKEND_DEFINE})

        if( GCL_MPI )
            set(ccflags ${MC_SPECIFIC_OPTIONS} )

            add_custom_mpi_mc_test(TARGET copy_stencil_parallel SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/./copy_stencil_parallel.cpp  ADDITIONAL_FLAGS ${ccflags})

            ## The next executable is not made into a test since it does not validate.
            ## The numerics need to be fixed (this is a task to be undertaken in the future)
            add_executable(shallow_water_enhanced_mc  shallow_water_enhanced.cpp)
            target_link_libraries(shallow_water_enhanced_mc  gtest_main gcl)
            target_compile_options(shallow_water_enhanced_mc PUBLIC ${GT_CXX_FLAGS} ${GT_CXX_HOST_ONLY_FLAGS} -D${MC_BACKEND_DEFINE})
        endif()
    endif(GT_ENABLE_TARGET_MC)

    if(GT_ENABLE_TARGET_CUDA)

       if(COMPILE_TO_PTX)
         CUDA_COMPILE_PTX(generated_files horizontal_diffusion.cu simple_hori_diff.cu copy_stencil.cu tridiagonal.cu boundary-condition.cu)
       else()
         foreach(srcfile IN LISTS PERFTEST_SRCFILES SRCFILES )
             add_executable( ${srcfile}_cuda ${srcfile}.cu)
             target_compile_options(${srcfile}_cuda PUBLIC ${GT_CXX_FLAGS} ${GT_CUDA_FLAGS} ${GPU_SPECIFIC_FLAGS} -D${CUDA_BACKEND_DEFINE})
             target_link_libraries(${srcfile}_cuda regression_fixture stencil-composition)

             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_cuda" "47 38 61")
             gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/${srcfile}_cuda" "63 38 10")
         endforeach(srcfile)
         foreach(perftest IN LISTS PERFTEST_SRCFILES)
             add_dependencies(perftests ${perftest}_cuda)
         endforeach()
       endif()

       if( GCL_MPI )
           add_custom_mpi_gpu_test(TARGET copy_stencil_parallel SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/copy_stencil_parallel.cu )

           ## The next executable is not made into a test since it does not validate.
           ## The numerics need to be fixed (this is a task to be undertaken in the future)
           add_executable(shallow_water_enhanced_cuda shallow_water_enhanced.cu)
           target_compile_options(shallow_water_enhanced_cuda PRIVATE ${GT_CUDA_FLAGS} ${GT_CXX_FLAGS} -D${CUDA_BACKEND_DEFINE})
           target_link_libraries(shallow_water_enhanced_cuda gtest_main gcl)
        endif()
    endif(GT_ENABLE_TARGET_CUDA)

    if ( COMPONENT_C_BINDINGS )
        add_subdirectory( c_bindings )
    endif()

    if ( COMPONENT_GCL )
        add_subdirectory( communication )
    endif()

  endif(NOT STRUCTURED_GRIDS)
endif( COMPONENT_STENCIL_COMPOSITION )
