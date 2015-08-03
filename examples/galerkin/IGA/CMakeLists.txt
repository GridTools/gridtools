add_executable( basis_function basis_functions.cpp)
set_target_properties(basis_function PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(basis_function ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

add_subdirectory( RunTimeIGA )
