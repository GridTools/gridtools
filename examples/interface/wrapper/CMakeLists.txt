
enable_language(Fortran)

foreach( repotype SIMPLE ) # REPOSITORY
  string(TOLOWER ${repotype} repo_suffix)
  
#  add_library( example_wrapper_${repo_suffix} c_interface_${repo_suffix}_wrapper.cpp)
  
  
#  if(CUDA_FOUND)
#    cuda_add_library( example_wrapper_${repo_suffix}_cuda c_interface_${repo_suffix}_wrapper.cu OPTIONS "${GPU_SPECIFIC_FLAGS}" )
#  endif()
  
  foreach( data_type FLOAT DOUBLE INT )
  foreach( dimension 2 3 )
    string(TOLOWER ${data_type} datatype_suffix)
    add_bindings_library(example_wrapper_${repo_suffix}_${datatype_suffix}_${dimension}D c_interface_${repo_suffix}_wrapper.cpp)
    target_link_libraries(example_wrapper_${repo_suffix}_${datatype_suffix}_${dimension}D gt_interface)
    set_target_properties(example_wrapper_${repo_suffix}_${datatype_suffix}_${dimension}D PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -DUSE_TYPE_${data_type} -DDIMENSION=${dimension}")
        
    foreach( variation NORMAL PADDING )
        string(TOLOWER ${variation} var_suffix)
        set( target_name "c_interface_${var_suffix}_${repo_suffix}_${datatype_suffix}_${dimension}D" )
        add_executable( ${target_name} c_interface_main.c )
        target_link_libraries( ${target_name} -Wl,--whole-archive example_wrapper_${repo_suffix}_${datatype_suffix}_${dimension}D -Wl,--no-whole-archive )
        set_target_properties( ${target_name} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -DC_INTERFACE_EXAMPLE_${variation} -DC_INTERFACE_EXAMPLE_${repotype} -DUSE_TYPE_${data_type} -DDIMENSION=${dimension} -std=c11")
        target_link_libraries( ${target_name} gt_interface )
        add_dependencies( ${target_name} example_wrapper_${repo_suffix}_${datatype_suffix}_${dimension}D_declarations)
        target_include_directories(${target_name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}") #include the generated bindings for the user defined wrapper
        gridtools_add_test( tests.${target_name} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${target_name}")
    #    if(CUDA_FOUND)
    #      string(TOLOWER ${variation} var_suffix)
    #      set( target_name_cuda "${target_name}_cuda" )
    #      add_executable( ${target_name_cuda} c_interface_main.c )
    #      set_target_properties( ${target_name_cuda} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -DC_INTERFACE_EXAMPLE_${variation} -DC_INTERFACE_EXAMPLE_${repotype} -DCUDA_EXAMPLE")
    #      target_link_libraries( ${target_name_cuda} -Wl,--whole-archive example_wrapper_${repo_suffix}_cuda -Wl,--no-whole-archive )
    #      target_link_libraries( ${target_name_cuda} gt_interface_cuda )
    #      target_link_libraries( ${target_name_cuda} ${CUDA_CUDART_LIBRARY} )
    #      target_link_libraries( ${target_name_cuda} "cuda" )
    #      gridtools_add_test( tests.${target_name_cuda} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/examples/${target_name}")
    #    endif()
    endforeach( variation )
    
    #SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DUSE_TYPE_${data_type} -DDIMENSION=${dimension}")
    add_executable( fortran_interface_simple_${datatype_suffix}_${dimension}D fortran_interface_simple.F90 wrapper_functions_delegate.F90 ${PROJECT_SOURCE_DIR}/src/interface/wrapper/gt_interface.f90 ${CMAKE_CURRENT_BINARY_DIR}/example_wrapper_simple.f90 ) #TODO make lib
    target_link_libraries( fortran_interface_simple_${datatype_suffix}_${dimension}D -Wl,--whole-archive example_wrapper_simple_${datatype_suffix}_${dimension}D -Wl,--no-whole-archive )
    set_target_properties(fortran_interface_simple_${datatype_suffix}_${dimension}D PROPERTIES COMPILE_FLAGS "${CMAKE_Fortran_FLAGS} -DUSE_TYPE_${data_type} -DTEST_DIMENSION=${dimension}")
  endforeach( dimension )
  endforeach( data_type )
endforeach( repotype )

