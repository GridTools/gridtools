macro(add_bindings_library) 
     
    if (DEFINED GRIDTOOLS_LIBRARIES_DIR)
        set(binding_libs ${GRIDTOOLS_LIBRARIES_DIR}/libc_bindings_generator.a
            ${GRIDTOOLS_LIBRARIES_DIR}/libc_bindings_handle.a)
        set(binding_f90_libs ${GRIDTOOLS_LIBRARIES_DIR}/libc_bindings_handle_fortran.a
            ${GRIDTOOLS_LIBRARIES_DIR}/libarray_descriptor.a)
        set(binding_main_lib  ${GRIDTOOLS_LIBRARIES_DIR}/libc_bindings_generator_main.a)
    else()
        set(binding_libs c_bindings_generator c_bindings_handle)
        set(binding_f90_libs c_bindings_handle_fortran array_descriptor)
        set(binding_main_lib c_bindings_generator_main)
    endif()

    set(target_name ${ARGV0})

    add_library(${ARGV})

    target_link_libraries(${target_name} ${binding_libs})
    add_custom_command(OUTPUT ${target_name}_empty.cpp COMMAND touch ${target_name}_empty.cpp)
    add_executable(${target_name}_decl_generator
            ${CMAKE_CURRENT_BINARY_DIR}/${target_name}_empty.cpp)
    set_target_properties(${target_name}_decl_generator PROPERTIES LINK_FLAGS -pthread)
    if (${APPLE})
          target_link_libraries(${target_name}_decl_generator
              -Wl,-force_load ${target_name} ${bindings_main}
              ${binding_main_lib})
    else()
          target_link_libraries(${target_name}_decl_generator
              -Xlinker --whole-archive ${target_name}
              -Xlinker --no-whole-archive ${binding_main_lib})
    endif()
    add_custom_command(OUTPUT ${target_name}.h ${target_name}.f90
            COMMAND ${target_name}_decl_generator ${target_name}.h ${target_name}.f90 ${target_name}
            DEPENDS $<TARGET_FILE:${target_name}_decl_generator>)
    add_custom_target(${target_name}_declarations
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.h ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.f90)

    add_library(${target_name}_c ${CMAKE_CURRENT_BINARY_DIR}/${target_name}_empty.cpp)
    target_link_libraries(${target_name}_c ${target_name})
    add_dependencies(${target_name}_c ${target_name}_declarations)
    target_include_directories(${target_name}_c INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")

    add_library(${target_name}_fortran ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.f90)
    target_link_libraries(${target_name}_fortran ${target_name} ${binding_f90_libs})
    add_dependencies(${target_name}_fortran ${target_name}_declarations)
    target_include_directories(${target_name}_fortran INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")

endmacro()

