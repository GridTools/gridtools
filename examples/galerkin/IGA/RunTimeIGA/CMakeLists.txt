add_executable( b_splines_eval_test_rt b_splines_eval_test_rt.cpp)
set_target_properties(b_splines_eval_test_rt PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
target_link_libraries(b_splines_eval_test_rt ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

add_executable( b_splines_basis_eval_test_rt b_splines_basis_eval_test_rt.cpp)
set_target_properties(b_splines_basis_eval_test_rt PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
target_link_libraries(b_splines_basis_eval_test_rt ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

add_executable( b_splines_surface_eval_test_rt b_splines_surface_eval_test_rt.cpp)
set_target_properties(b_splines_surface_eval_test_rt PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
target_link_libraries(b_splines_surface_eval_test_rt ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

add_executable( iga_test_1_rt iga_test_1_rt.cpp)
set_target_properties(iga_test_1_rt PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
target_link_libraries(iga_test_1_rt ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

add_executable( iga_test_2_rt iga_test_2_rt.cpp)
set_target_properties(iga_test_2_rt PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
target_link_libraries(iga_test_2_rt ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})

add_executable( iga_test_3_rt iga_test_3_rt.cpp)
set_target_properties(iga_test_3_rt PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
target_link_libraries(iga_test_3_rt ${exe_LIBS} gcl ${MPI_ADDITIONAL_LIBS})
