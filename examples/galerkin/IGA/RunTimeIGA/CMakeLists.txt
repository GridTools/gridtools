add_executable( b_splines_eval_test_rt b_splines_eval_test_rt.cpp)
set_target_properties(b_splines_eval_test_rt PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(b_splines_eval_test_rt ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
