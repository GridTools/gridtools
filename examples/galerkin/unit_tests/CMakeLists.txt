add_executable( test_legendre legendre.cpp )
set_target_properties( test_legendre PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} )
target_link_libraries( test_legendre ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS}  ${INTREPID_LIB}  ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES}  ${LAPACK_LIBRARIES} ${BLAS_LIBS} )

add_executable( test_interpolation interpolation.cpp )
set_target_properties( test_interpolation PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} )
target_link_libraries( test_interpolation ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS}  ${INTREPID_LIB}  ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES}  ${LAPACK_LIBRARIES} ${BLAS_LIBS} )

cuda_add_executable( test_interpolation_cuda interpolation.cu )
set_target_properties( test_interpolation_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} )
target_link_libraries( test_interpolation_cuda ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS}  ${INTREPID_LIB}  ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES}  ${LAPACK_LIBRARIES} ${BLAS_LIBS} )

cuda_add_executable( test_legendre_cuda legendre.cu )
set_target_properties( test_legendre_cuda PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} )
target_link_libraries( test_legendre_cuda ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS}  ${INTREPID_LIB}  ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES}  ${LAPACK_LIBRARIES} ${BLAS_LIBS} )
