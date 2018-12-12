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
      target_link_libraries(${srcfile} gtest gcl mpi_gtest_main )
      target_compile_definitions(${srcfile} PRIVATE STANDALONE)
    endforeach(srcfile)


    if( GT_ENABLE_TARGET_CUDA )
      if(NOT MSVC)
        foreach(srcfile
                test_halo_exchange_3D_all
                test_halo_exchange_3D_all_2
                test_halo_exchange_3D_all_3
                test_halo_exchange_3D_generic
                test_halo_exchange_3D_generic_full
                )
          add_executable( ${srcfile}_cu ${srcfile}.cu )
          target_link_libraries( ${srcfile}_cu gtest gcl mpi_gtest_main GridToolsTestCUDA)
          target_compile_definitions( ${srcfile}_cu PRIVATE STANDALONE)
        endforeach()
      endif()
    endif()
  endif()
endif()
