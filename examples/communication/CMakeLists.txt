set( OLD_OPTIONS "${CMAKE_CXX_FLAGS}" )


### Executables
if(GCL_ONLY)
else(GCL_ONLY)
###   message("${HOST_SPECIFIC_OPTIONS}")
###   message("${CMAKE_CXX_FLAGS}")
###   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HOST_SPECIFIC_OPTIONS}" )
###   message("${CMAKE_CXX_FLAGS}")
  ### L3
  if( GCL_MPI )
    #CPU 2D MPI tests
    foreach(srcfile test_generic_all_to_all )
      add_executable( ${srcfile} ../../examples/communication/${srcfile}.cpp)
      add_dependencies(${srcfile} gcl)
          set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS}")
      target_link_libraries(${srcfile} gcl ${MPI_ADDITIONAL_LIBS} ${CUDA_LIBRARIES})
    endforeach(srcfile)
  endif()

  ### L2
  if( GCL_MPI )
    #CPU 2D MPI tests
    foreach(srcfile test_async_progress_3D_generic test_all_to_all_halo test_all_to_all_ij_ji test_all_to_all_halo_3D test_halo_exchange test_halo_exchange_2D test_halo_exchange_3D test_halo_exchange_3D_all test_halo_exchange_3D_all_2 test_halo_exchange_3D_all_3 test_halo_exchange_3D_generic test_halo_exchange_2D_generic test_halo_exchange_3D_generic_full)
      add_executable( ${srcfile} ../../examples/communication/${srcfile}.cpp)
      add_dependencies(${srcfile} gcl)
      set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS}")
      target_link_libraries(${srcfile} gcl ${MPI_ADDITIONAL_LIBS} ${CUDA_LIBRARIES})
    endforeach(srcfile)
  endif()


  if( GCL_GPU )
    ##set(CMAKE_CXX_FLAGS "${OLD_OPTIONS}")
    if( GCL_MPI )
      if ( USE_MPI_COMPILER )
        #message( "MPI enabled for CUDA - compiling with mpi compiler" )
        include_directories( ${MPI_INCLUDE_PATH} )
        #message( "MPI include path: ${MPI_INCLUDE_PATH}" )
        #link_directories( ${MPI_LIBRARY} )
      else( USE_MPI_COMPILER )
        #message( "MPI enabled for CUDA - compiling with wrappers" )
        set( MPI_INCLUDE_PATH $ENV{MPICH_DIR}/include )
        #set( MPI_LIBRARY_PATH $ENV{MPICH_DIR}/lib )
        #set( CUDA_MPI_LIB mpi )
        #link_directories( ${MPI_LIBRARY_PATH} )
        include_directories( ${MPI_INCLUDE_PATH} )
        #message( "MPI include path: ${MPI_INCLUDE_PATH}" )
      endif( USE_MPI_COMPILER )
    endif( GCL_MPI )
    if( GCL_MPI )
      if(NOT MSVC)
        cuda_add_executable( cuda_test_halo_exchange_3D_all ../../examples/communication/test_halo_exchange_3D_all.cu )
        add_dependencies( cuda_test_halo_exchange_3D_all gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_all_2 ../../examples/communication/test_halo_exchange_3D_all_2.cu )
        add_dependencies( cuda_test_halo_exchange_3D_all_2 gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all_2 gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_all_3 ../../examples/communication/test_halo_exchange_3D_all_3.cu )
        add_dependencies( cuda_test_halo_exchange_3D_all_3 gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all_3 gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_generic ../../examples/communication/test_halo_exchange_3D_generic.cu )
        add_dependencies( cuda_test_halo_exchange_3D_generic gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_generic gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_async_progress_3D_generic ../../examples/communication/test_async_progress_3D_generic.cu )
        add_dependencies( cuda_test_async_progress_3D_generic gcl )
        target_link_libraries( cuda_test_async_progress_3D_generic gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_generic_full ../../examples/communication/test_halo_exchange_3D_generic_full.cu )
        add_dependencies( cuda_test_halo_exchange_3D_generic_full gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_generic_full gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_2D_generic ../../examples/communication/test_halo_exchange_2D_generic.cu )
        add_dependencies( cuda_test_halo_exchange_2D_generic gcl )
        target_link_libraries( cuda_test_halo_exchange_2D_generic gcl ${CUDA_MPI_LIB} )

      endif()
    endif()
  endif()
endif()
