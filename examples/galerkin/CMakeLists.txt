add_executable( assembly assembly.cpp)
set_target_properties(assembly PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(assembly ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
