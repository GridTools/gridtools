set(CUDA_SEPARABLE_COMPILATION OFF)

### Executables
if(GCL_ONLY)
else(GCL_ONLY)

  ### L2
  if( USE_MPI )
    foreach(srcfile test_all_to_all_halo_3D test_halo_exchange_3D_all test_halo_exchange_3D_all_2 test_halo_exchange_3D_all_3 test_halo_exchange_3D_generic test_halo_exchange_3D_generic_full)
      add_executable( ${srcfile} ../../regression/communication/${srcfile}.cpp)
      add_dependencies(${srcfile} gcl)
      set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DSTANDALONE")
      target_link_libraries(${srcfile} ${exe_LIBS} )
    endforeach(srcfile)
  endif()


  if( ENABLE_CUDA )
    if( GCL_MPI )
      include_directories( ${MPI_INCLUDE_PATH} )
    endif( GCL_MPI )
    if( USE_MPI )
      if(NOT MSVC)
        cuda_add_executable( cuda_test_halo_exchange_3D_all ../../regression/communication/test_halo_exchange_3D_all.cu OPTIONS -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        add_dependencies( cuda_test_halo_exchange_3D_all gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all   ${exe_LIBS} )

        cuda_add_executable( cuda_test_halo_exchange_3D_all_2 ../../regression/communication/test_halo_exchange_3D_all_2.cu  OPTIONS -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        add_dependencies( cuda_test_halo_exchange_3D_all_2 gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all_2   ${exe_LIBS} )

        cuda_add_executable( cuda_test_halo_exchange_3D_all_3 ../../regression/communication/test_halo_exchange_3D_all_3.cu  OPTIONS -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        add_dependencies( cuda_test_halo_exchange_3D_all_3 gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all_3   ${exe_LIBS} )

        cuda_add_executable( cuda_test_halo_exchange_3D_generic ../../regression/communication/test_halo_exchange_3D_generic.cu  OPTIONS -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        add_dependencies( cuda_test_halo_exchange_3D_generic gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_generic  ${exe_LIBS} )

        cuda_add_executable( cuda_test_halo_exchange_3D_generic_full ../../regression/communication/test_halo_exchange_3D_generic_full.cu  OPTIONS -DSTANDALONE ${GPU_SPECIFIC_FLAGS})
        add_dependencies( cuda_test_halo_exchange_3D_generic_full gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_generic_full  ${exe_LIBS} )

      endif()
    endif()
  endif()
endif()
