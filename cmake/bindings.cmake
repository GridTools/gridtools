macro(add_bindings_library)
    add_library(${ARGV})
    target_link_libraries(${ARGV0} c_bindings_generator)
    add_custom_command(OUTPUT ${ARGV0}_empty.cpp COMMAND touch ${ARGV0}_empty.cpp)
    add_executable(${ARGV0}_decl_generator ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}_empty.cpp)
    if (${APPLE})
        target_link_libraries(${ARGV0}_decl_generator -force_load ${ARGV0} c_bindings_generator_main)
    else()
        target_link_libraries(${ARGV0}_decl_generator -Wl,--whole-archive ${ARGV0} -Wl,--no-whole-archive
                c_bindings_generator_main)
    endif()
    add_custom_command(OUTPUT ${ARGV0}.h ${ARGV0}.f90
            COMMAND ${ARGV0}_decl_generator ${ARGV0}.h ${ARGV0}.f90
            DEPENDS $<TARGET_FILE:${ARGV0}_decl_generator>)
    add_custom_target(${ARGV0}_decl
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}.h ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}.f90)
endmacro()