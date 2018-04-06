macro(add_bindings_library)
    add_library(${ARGV})
    target_link_libraries(${ARGV0} c_bindings_generator c_bindings_handle)
    add_custom_command(OUTPUT ${ARGV0}_empty.cpp COMMAND touch ${ARGV0}_empty.cpp)
    add_executable(${ARGV0}_decl_generator ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}_empty.cpp)
    if (${APPLE})
        target_link_libraries(${ARGV0}_decl_generator -Wl,-force_load ${ARGV0} c_bindings_generator_main)
    else()
        target_link_libraries(${ARGV0}_decl_generator -Wl,--whole-archive ${ARGV0} -Wl,--no-whole-archive
                c_bindings_generator_main)
    endif()
    add_custom_command(OUTPUT ${ARGV0}.h ${ARGV0}.f90
            COMMAND ${ARGV0}_decl_generator ${ARGV0}.h ${ARGV0}.f90 ${ARGV0}
            DEPENDS $<TARGET_FILE:${ARGV0}_decl_generator>)
    add_custom_target(${ARGV0}_declarations
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}.h ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}.f90)

    add_library(${ARGV0}_c ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}_empty.cpp)
    target_link_libraries(${ARGV0}_c ${ARGV0})
    add_dependencies(${ARGV0}_c ${ARGV0}_declarations)
    target_include_directories(${ARGV0}_c INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")

    add_library(${ARGV0}_fortran ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}.f90)
    target_link_libraries(${ARGV0}_fortran ${ARGV0} c_bindings_handle_fortran)
    add_dependencies(${ARGV0}_fortran ${ARGV0}_declarations)
    target_include_directories(${ARGV0}_fortran INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")

endmacro()
