FILE(GLOB_RECURSE INC_ALL "*.hpp") 

add_executable( main_example main.cpp ${INC_ALL})
set_target_properties(main_example PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(main_example ${exe_LIBS})

