#####################look for Xdmf library########################
set(XDMF_ROOT "OFF" CACHE PATH "location of the xdmf library")
if(ENABLE_XDMF)
  find_package(Xdmf REQUIRED HINTS ${XDMF_ROOT} "${CMAKE_SOURCE_DIR}/cmake/modules")
  if(XDMF_FOUND)
    include_directories( "${XDMF_INCLUDE_DIRS}" )
    set ( exe_LIBS "${exe_LIBS}" "${XDMF_LIBRARIES}" )
    message("XDMF libs found in " "${XDMF_LIBRARIES}")
    message("XDMF includes found in " "${XDMF_INCLUDE_DIRS}")
  else()
    if(XDMF_LIBS AND XDMF_INCLUDE_DIRS)
      set ( exe_LIBS "${exe_LIBS}" "${XDMF_LIBS}" )
      message("xdmf libs: " "${XDMF_LIBS}")
      include_directories( "${XDMF_INCLUDE_DIRS}" )
    else()
      message("Xdmf library not found. set the XDMF_INCLUDE_DIR and XDMF_LIBS to the headers and libraries path.")
    endif(XDMF_LIBS AND XDMF_INCLUDE_DIRS)
  endif(XDMF_FOUND)
endif(ENABLE_XDMF)
##################################################################

include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )

################### look for intrepid, teuchos, blas, lapack ############################
set( ENABLE_INTREPID "OFF" CACHE BOOL "Enable support of finite elements discretization from the Intrepid library" )
if(ENABLE_INTREPID)

  if(NOT INTREPID_SOURCE_DIR)
    message("set the INTREPID_SOURCE_DIR to point to the intrepid sources")
  else()
    message("intrepid headers in " ${INTREPID_SOURCE_DIR})
    include_directories( SYSTEM ${INTREPID_SOURCE_DIR} )
  endif(NOT INTREPID_SOURCE_DIR)
  if(NOT INTREPID_LIB_DIR)
    message("set the INTREPID_LIB_DIR to point to the intrepid libraries")
  else()
    find_library( INTREPID_LIB intrepid ${INTREPID_LIB_DIR})
    message("intrepid found in " ${INTREPID_LIB})

    find_library( KOKKOS_LIB kokkoscore ${INTREPID_LIB_DIR})
    message("kokkos found in " ${KOKKOS_LIB})

    find_library( TEUCHOS_LIB teuchoscore ${INTREPID_LIB_DIR})
    message("teuchos found in " ${TEUCHOS_LIB})

    find_library( TEUCHOS_NUMERICS_LIB teuchosnumerics ${INTREPID_LIB_DIR})
    message("teuchos numerics found in " ${TEUCHOS_NUMERICS_LIB})

    find_library( TEUCHOS_KOKKOS_COMM_LIB teuchoskokkoscomm ${INTREPID_LIB_DIR})
    message("teuchos kokkos comms found in " ${TEUCHOS_KOKKOS_COMM_LIB})

    find_library( TEUCHOS_KOKKOS_COMPAT_LIB teuchoskokkoscompat ${INTREPID_LIB_DIR})
    message("teuchos kokkos compat found in " ${TEUCHOS_KOKKOS_COMPAT_LIB})

    find_library( TEUCHOS_PARAMETER_LIST_LIB teuchosparameterlist ${INTREPID_LIB_DIR})
    message("teuchos parameter list found in " ${TEUCHOS_PARAMETER_LIST_LIB})

    find_library( TEUCHOS_REMAINDER_LIB teuchosremainder ${INTREPID_LIB_DIR})
    message("teuchos remainder found in " ${TEUCHOS_REMAINDER_LIB})

    find_library( TEUCHOS_COMM_LIB teuchoscomm ${INTREPID_LIB_DIR})
    message("teuchos numerics found in " ${TEUCHOS_COMM_LIB})

    find_library( SHARDS_LIB shards ${INTREPID_LIB_DIR})
    message("shards found in " ${SHARDS_LIB})

    if(BLAS_DIR)
        message("looking for BLAS in" ${BLAS_DIR})
        set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${BLAS_DIR})
    else()
        message("BLAS not found. Set the BLAS_DIR manually")
    endif(BLAS_DIR)

    if(LAPACK_DIR)
        message("-- looking for LAPACK in" ${LAPACK_DIR})
          set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${LAPACK_DIR})
    else()
        message("LAPACK not found. Set the LAPACK_DIR manually")
    endif(LAPACK_DIR)

   find_package(BLAS)
   set( BLAS_LIBS  ${BLAS_LIBRARIES})

   if(NOT BLAS_FOUND)
          message("BLAS library not found. Set the path to the installation via the BLAS_DIR CMake variable.")
          if(BLAS_HARDCODED_LIB)
          set( BLAS_LIBS  ${BLAS_HARDCODED_LIB})
          message ("setting BLAS library to an hardcoded path: " ${BLAS_HARDCODED_LIB})
          endif()
   endif(NOT BLAS_FOUND)

   find_package(LAPACK)
   if(NOT LAPACK_FOUND)
          message("Lapack library not found. Set the path to the installation via the LAPACK_DIR CMake variable, or hardcode it by setting LAPACK_HARDCODED_LIB to the library.")
   endif(NOT LAPACK_FOUND)

   if(LAPACK_HARDCODED_LIB)
     set( LAPACK_LIBRARIES ${LAPACK_HARDCODED_LIB})
     message ("setting LAPACK library to an hardcoded path: " ${LAPACK_HARDCODED_LIB})
   endif()

   message("Blas in " ${BLAS_LIBS})
   message("Lapack in " ${LAPACK_LIBRARIES})
   message("Teuchos Numerics in " ${TEUCHOS_LIB})

endif(NOT INTREPID_LIB_DIR)
endif(ENABLE_INTREPID)
######################################################################

######################## add the example executables #############
list( APPEND EXAMPLES stiffness mass mass_2D mass_IGA dg_flux gather dg_advection legendre linear_solver)
foreach(example IN LISTS EXAMPLES)
add_executable( ${example} examples/${example}.cpp )
set_target_properties( ${example} PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS} )
target_link_libraries( ${example} ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS}  ${INTREPID_LIB}  ${SHARDS_LIB} ${TEUCHOS_COMM_LIB} ${TEUCHOS_PARAMETER_LIST_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${KOKKOS_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBS} )
endforeach(example)
##################################################################

#TODO: remove this test?
add_executable( legendre2 examples/legendre/legendre.cpp examples/legendre/main.cpp examples/legendre/mesh.cpp examples/legendre/compute_jacobian.cpp examples/legendre/compute_assembly.cpp)
set_target_properties( legendre2 PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS} )
target_link_libraries( legendre2 ${exe_LIBS}  ${INTREPID_LIB} ${SHARDS_LIB}  ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBS})

if(CUDA_FOUND)
########################### add cuda executables #####################
list( APPEND EXAMPLES_CUDA stiffness mass dg_flux dg_advection legendre mass_IGA)
foreach(example IN LISTS EXAMPLES_CUDA)
cuda_add_executable( ${example}_cuda examples/${example}.cu )
target_link_libraries( ${example}_cuda ${exe_LIBS} ${INTREPID_LIB} ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBS} )
endforeach(example)
########################################################################################

cuda_add_executable( legendre2_cuda examples/legendre/compute_assembly.cu examples/legendre/compute_jacobian.cu examples/legendre/main.cu examples/legendre/legendre.cu examples/legendre/mesh.cu )
target_link_libraries(legendre2_cuda ${exe_LIBS} ${INTREPID_LIB} ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBS} )

endif(CUDA_FOUND)

add_subdirectory( IGA )
add_subdirectory( doc )
add_subdirectory( unit_tests )
