set( OLD_OPTIONS "${CMAKE_CXX_FLAGS}" )
set(CUDA_SEPARABLE_COMPILATION OFF)

### Executables
if(GCL_ONLY)
else(GCL_ONLY)
###   message("${HOST_SPECIFIC_OPTIONS}")
###   message("${CMAKE_CXX_FLAGS}")
###   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HOST_SPECIFIC_OPTIONS}" )
###   message("${CMAKE_CXX_FLAGS}")

  ### L2
  if( GCL_MPI )
    #CPU 2D MPI tests
    foreach(srcfile test_all_to_all_halo_3D test_halo_exchange_3D_all test_halo_exchange_3D_all_2 test_halo_exchange_3D_all_3 test_halo_exchange_3D_generic test_halo_exchange_3D_generic_full)
      add_executable( ${srcfile} ../../examples/communication/${srcfile}.cpp)
      add_dependencies(${srcfile} gcl)
      set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DSTANDALONE")
      target_link_libraries(${srcfile} gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_LIBRARIES})
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
        cuda_add_executable( cuda_test_halo_exchange_3D_all ../../examples/communication/test_halo_exchange_3D_all.cu OPTIONS -DSTANDALONE)
        add_dependencies( cuda_test_halo_exchange_3D_all gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all gcl  ${Boost_LIBRARIES} ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_all_2 ../../examples/communication/test_halo_exchange_3D_all_2.cu  OPTIONS -DSTANDALONE)
        add_dependencies( cuda_test_halo_exchange_3D_all_2 gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all_2 gcl  ${Boost_LIBRARIES} ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_all_3 ../../examples/communication/test_halo_exchange_3D_all_3.cu  OPTIONS -DSTANDALONE)
        add_dependencies( cuda_test_halo_exchange_3D_all_3 gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_all_3 gcl  ${Boost_LIBRARIES} ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_generic ../../examples/communication/test_halo_exchange_3D_generic.cu  OPTIONS -DSTANDALONE)
        add_dependencies( cuda_test_halo_exchange_3D_generic gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_generic gcl ${Boost_LIBRARIES} ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_generic_full ../../examples/communication/test_halo_exchange_3D_generic_full.cu  OPTIONS -DSTANDALONE)
        add_dependencies( cuda_test_halo_exchange_3D_generic_full gcl )
        target_link_libraries( cuda_test_halo_exchange_3D_generic_full gcl ${Boost_LIBRARIES} ${CUDA_MPI_LIB} )

      endif()
    endif()
  endif()
endif()

  if( GCL_MPI )
    if ( $ENV{JENKINS_COMMUNICATION_TESTS} )

################################## GPU
    if( GCL_GPU )
        cuda_add_executable( test_halo_exchange_3D_cuda_generic_0 ../../examples/communication/test_halo_exchange_3D_generic.cu)
        add_dependencies( test_halo_exchange_3D_cuda_generic_0 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_generic_0 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_generic_0 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_generic_1 ../../examples/communication/test_halo_exchange_3D_generic.cu)
        add_dependencies( test_halo_exchange_3D_cuda_generic_1 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_generic_1 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_generic_1 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_generic_2 ../../examples/communication/test_halo_exchange_3D_generic.cu)
        add_dependencies( test_halo_exchange_3D_cuda_generic_2 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_generic_2 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_generic_2 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_generic_3 ../../examples/communication/test_halo_exchange_3D_generic.cu)
        add_dependencies( test_halo_exchange_3D_cuda_generic_3 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_generic_3 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_generic_3 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_generic_4 ../../examples/communication/test_halo_exchange_3D_generic.cu)
        add_dependencies( test_halo_exchange_3D_cuda_generic_4 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_generic_4 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_generic_4 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_generic_5 ../../examples/communication/test_halo_exchange_3D_generic.cu)
        add_dependencies( test_halo_exchange_3D_cuda_generic_5 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_generic_5 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_generic_5 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_generic_6 ../../examples/communication/test_halo_exchange_3D_generic.cu)
        add_dependencies( test_halo_exchange_3D_cuda_generic_6 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_generic_6 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_generic_6 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_generic_7 ../../examples/communication/test_halo_exchange_3D_generic.cu)
        add_dependencies( test_halo_exchange_3D_cuda_generic_7 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_generic_7 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_generic_7 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_0 ../../examples/communication/test_halo_exchange_3D_all.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_0 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_0 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_0 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_1 ../../examples/communication/test_halo_exchange_3D_all.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_1 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_1 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_1 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2 ../../examples/communication/test_halo_exchange_3D_all.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_3 ../../examples/communication/test_halo_exchange_3D_all.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_3 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_3 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_3 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_4 ../../examples/communication/test_halo_exchange_3D_all.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_4 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_4 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_4 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_5 ../../examples/communication/test_halo_exchange_3D_all.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_5 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_5 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_5 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_6 ../../examples/communication/test_halo_exchange_3D_all.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_6 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_6 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_6 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_7 ../../examples/communication/test_halo_exchange_3D_all.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_7 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_7 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_7 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2_0 ../../examples/communication/test_halo_exchange_3D_all_2.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2_0 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2_0 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2_0 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2_1 ../../examples/communication/test_halo_exchange_3D_all_2.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2_1 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2_1 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2_1 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2_2 ../../examples/communication/test_halo_exchange_3D_all_2.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2_2 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2_2 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2_2 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2_3 ../../examples/communication/test_halo_exchange_3D_all_2.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2_3 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2_3 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2_3 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2_4 ../../examples/communication/test_halo_exchange_3D_all_2.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2_4 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2_4 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2_4 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2_5 ../../examples/communication/test_halo_exchange_3D_all_2.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2_5 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2_5 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2_5 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2_6 ../../examples/communication/test_halo_exchange_3D_all_2.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2_6 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2_6 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2_6 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

        cuda_add_executable( test_halo_exchange_3D_cuda_all_test_2_7 ../../examples/communication/test_halo_exchange_3D_all_2.cu)
        add_dependencies( test_halo_exchange_3D_cuda_all_test_2_7 gcl)
        set_target_properties( test_halo_exchange_3D_cuda_all_test_2_7 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_cuda_all_test_2_7 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} ${CUDA_MPI_LIB} ${CUDA_LIBRARIES})

     else()

        add_executable( test_halo_exchange_3D_generic_0 ../../examples/communication/test_halo_exchange_3D_generic.cpp)
        add_dependencies( test_halo_exchange_3D_generic_0 gcl)
        set_target_properties( test_halo_exchange_3D_generic_0 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -std=c++11")
        target_link_libraries( test_halo_exchange_3D_generic_0 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_generic_1 ../../examples/communication/test_halo_exchange_3D_generic.cpp)
        add_dependencies( test_halo_exchange_3D_generic_1 gcl)
        set_target_properties( test_halo_exchange_3D_generic_1 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -std=c++11")
        target_link_libraries( test_halo_exchange_3D_generic_1 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_generic_2 ../../examples/communication/test_halo_exchange_3D_generic.cpp)
        add_dependencies( test_halo_exchange_3D_generic_2 gcl)
        set_target_properties( test_halo_exchange_3D_generic_2 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_generic_2 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_generic_3 ../../examples/communication/test_halo_exchange_3D_generic.cpp)
        add_dependencies( test_halo_exchange_3D_generic_3 gcl)
        set_target_properties( test_halo_exchange_3D_generic_3 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_generic_3 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_generic_4 ../../examples/communication/test_halo_exchange_3D_generic.cpp)
        add_dependencies( test_halo_exchange_3D_generic_4 gcl)
        set_target_properties( test_halo_exchange_3D_generic_4 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_generic_4 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_generic_5 ../../examples/communication/test_halo_exchange_3D_generic.cpp)
        add_dependencies( test_halo_exchange_3D_generic_5 gcl)
        set_target_properties( test_halo_exchange_3D_generic_5 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_generic_5 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_generic_6 ../../examples/communication/test_halo_exchange_3D_generic.cpp)
        add_dependencies( test_halo_exchange_3D_generic_6 gcl)
        set_target_properties( test_halo_exchange_3D_generic_6 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_generic_6 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_generic_7 ../../examples/communication/test_halo_exchange_3D_generic.cpp)
        add_dependencies( test_halo_exchange_3D_generic_7 gcl)
        set_target_properties( test_halo_exchange_3D_generic_7 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_generic_7 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_0 ../../examples/communication/test_halo_exchange_3D_all.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_0 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_0 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -std=c++11")
        target_link_libraries( test_halo_exchange_3D_all_test_0 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_1 ../../examples/communication/test_halo_exchange_3D_all.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_1 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_1 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -std=c++11")
        target_link_libraries( test_halo_exchange_3D_all_test_1 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2 ../../examples/communication/test_halo_exchange_3D_all.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_all_test_2 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_3 ../../examples/communication/test_halo_exchange_3D_all.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_3 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_3 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_all_test_3 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_4 ../../examples/communication/test_halo_exchange_3D_all.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_4 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_4 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_all_test_4 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_5 ../../examples/communication/test_halo_exchange_3D_all.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_5 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_5 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_all_test_5 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_6 ../../examples/communication/test_halo_exchange_3D_all.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_6 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_6 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_all_test_6 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_7 ../../examples/communication/test_halo_exchange_3D_all.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_7 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_7 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_all_test_7 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2_0 ../../examples/communication/test_halo_exchange_3D_all_2.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2_0 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2_0 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -std=c++11")
        target_link_libraries( test_halo_exchange_3D_all_test_2_0 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2_1 ../../examples/communication/test_halo_exchange_3D_all_2.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2_1 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2_1 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -std=c++11")
        target_link_libraries( test_halo_exchange_3D_all_test_2_1 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2_2 ../../examples/communication/test_halo_exchange_3D_all_2.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2_2 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2_2 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_all_test_2_2 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2_3 ../../examples/communication/test_halo_exchange_3D_all_2.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2_3 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2_3 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -std=c++11")
        target_link_libraries( test_halo_exchange_3D_all_test_2_3 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2_4 ../../examples/communication/test_halo_exchange_3D_all_2.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2_4 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2_4 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_all_test_2_4 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2_5 ../../examples/communication/test_halo_exchange_3D_all_2.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2_5 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2_5 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_all_test_2_5 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2_6 ../../examples/communication/test_halo_exchange_3D_all_2.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2_6 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2_6 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_all_test_2_6 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )

        add_executable( test_halo_exchange_3D_all_test_2_7 ../../examples/communication/test_halo_exchange_3D_all_2.cpp)
        add_dependencies( test_halo_exchange_3D_all_test_2_7 gcl)
        set_target_properties( test_halo_exchange_3D_all_test_2_7 PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS} -DVECTOR_INT -DPACKING_TYPE=gridtools::version_mpi_pack -DCXX11_DISABLE")
        target_link_libraries( test_halo_exchange_3D_all_test_2_7 gcl  ${Boost_LIBRARIES} ${MPI_ADDITIONAL_LIBS} )


      endif()

    endif()
  endif()
