add_executable( test_legendre legendre.cpp )
set_target_properties( test_legendre PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} )
target_link_libraries( test_legendre ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS}  ${INTREPID_LIB}  ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES}  ${LAPACK_LIBRARIES} ${BLAS_LIBS} )

add_executable( test_interpolation interpolation.cpp )
set_target_properties( test_interpolation PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} )
target_link_libraries( test_interpolation ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS}  ${INTREPID_LIB}  ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES}  ${LAPACK_LIBRARIES} ${BLAS_LIBS} )

add_executable( cg_linear_solver cg_linear_solver.cpp )
set_target_properties( cg_linear_solver PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS} )
target_link_libraries( cg_linear_solver ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS} ${INTREPID_LIB}  ${SHARDS_LIB} ${TEUCHOS_NUMERICS_LIB} ${TEUCHOS_LIB} ${LAPACK_LIBRARIES} )

