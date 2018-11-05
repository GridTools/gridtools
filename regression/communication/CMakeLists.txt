### Executables
if(NOT GT_GCL_ONLY)

  ### L2
  if( GT_USE_MPI )
    foreach(srcfile
            test_all_to_all_halo_3D
            test_halo_exchange_3D_all
            test_halo_exchange_3D_all_2
            test_halo_exchange_3D_all_3
            test_halo_exchange_3D_generic
            test_halo_exchange_3D_generic_full
            )
      add_executable( ${srcfile} ${srcfile}.cpp)
      target_compile_options(${srcfile} PUBLIC ${HOST_SPECIFIC_OPTIONS} -DSTANDALONE)
      target_link_libraries(${srcfile} gtest gcl mpi_gtest_main )
    endforeach(srcfile)


    if( GT_ENABLE_CUDA )
      if(NOT MSVC)
        add_executable( cuda_test_halo_exchange_3D_all test_halo_exchange_3D_all.cu )
        target_compile_options( cuda_test_halo_exchange_3D_all PUBLIC ${GT_CUDA_FLAGS} ${GT_CXX_FLAGS} -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        target_link_libraries( cuda_test_halo_exchange_3D_all   gtest gcl mpi_gtest_main )

        add_executable( cuda_test_halo_exchange_3D_all_2 test_halo_exchange_3D_all_2.cu )
        target_compile_options( cuda_test_halo_exchange_3D_all_2 PUBLIC ${GT_CUDA_FLAGS} ${GT_CXX_FLAGS} -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        target_link_libraries( cuda_test_halo_exchange_3D_all_2   gtest gcl mpi_gtest_main )

        add_executable( cuda_test_halo_exchange_3D_all_3 test_halo_exchange_3D_all_3.cu )
        target_compile_options( cuda_test_halo_exchange_3D_all_3 PUBLIC ${GT_CUDA_FLAGS} ${GT_CXX_FLAGS} -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        target_link_libraries( cuda_test_halo_exchange_3D_all_3   gtest gcl mpi_gtest_main )

        add_executable( cuda_test_halo_exchange_3D_generic test_halo_exchange_3D_generic.cu )
        target_compile_options( cuda_test_halo_exchange_3D_generic PUBLIC ${GT_CUDA_FLAGS} ${GT_CXX_FLAGS} -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        target_link_libraries( cuda_test_halo_exchange_3D_generic  gtest gcl mpi_gtest_main )

        add_executable( cuda_test_halo_exchange_3D_generic_full test_halo_exchange_3D_generic_full.cu )
        target_compile_options( cuda_test_halo_exchange_3D_generic_full PUBLIC ${GT_CUDA_FLAGS} ${GT_CXX_FLAGS} -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        target_link_libraries( cuda_test_halo_exchange_3D_generic_full  gtest gcl mpi_gtest_main )

      endif()
    endif()
  endif()
endif()
