
add_bindings_library(mini_dycore_bindings mini_dycore.cpp)
target_link_libraries(mini_dycore_bindings gt_interface) # bindings providing push and pull for wrappables

add_executable(mini_dycore main.F90 mini_dycore_bindings.f90)
target_link_libraries(mini_dycore mini_dycore_bindings) # user-defined bindings
target_link_libraries(mini_dycore gt_interface_fortran) # bindings providing push and pull for wrappables
target_include_directories(mini_dycore PRIVATE ${CMAKE_CURRENT_BINARY_DIR}) # add build dir since they include the generated files
add_dependencies(mini_dycore mini_dycore_bindings_declarations)

gridtools_add_test( tests.mini_dycore ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/interface/repository_wrapper/mini_dycore")
