function(check_and_update target_file new_file)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E compare_files ${target_file} ${new_file}
        RESULT_VARIABLE compare_result
        )

    if(${compare_result} EQUAL 0)
        message(STATUS "${target_file} is up-to-date")
    else()
        message(WARNING "${target_file} was generated! "
            "If you ship the generated bindings with your sources, don't forget to ship this updated file (and its variants). "
            "Otherwise, this warning can be ignored.")

        get_filename_component(target_path ${target_file} PATH)
        file(COPY ${new_file} DESTINATION ${target_path})
    endif()
    file(REMOVE ${new_file})
endfunction()

# Generate bindings and compare against existing ones
message(STATUS "Generating bindings for library ${FORTRAN_MODULE_NAME}")

set(generator_dir ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/generated_bindings)
file(MAKE_DIRECTORY ${generator_dir})
get_filename_component(filename_BINDINGS_C_DECL_FILENAME ${BINDINGS_C_DECL_FILENAME} NAME)
set(new_BINDINGS_C_DECL_FILENAME ${generator_dir}/${filename_BINDINGS_C_DECL_FILENAME})
get_filename_component(filename_BINDINGS_FORTRAN_DECL_FILENAME ${BINDINGS_FORTRAN_DECL_FILENAME} NAME)
set(new_BINDINGS_FORTRAN_DECL_FILENAME ${generator_dir}/${filename_BINDINGS_FORTRAN_DECL_FILENAME})

# run generator
execute_process(COMMAND ${GENERATOR} ${new_BINDINGS_C_DECL_FILENAME} ${new_BINDINGS_FORTRAN_DECL_FILENAME} ${FORTRAN_MODULE_NAME}
    RESULT_VARIABLE generate_result
    OUTPUT_VARIABLE generate_out
    ERROR_VARIABLE generate_out
    )

if(${generate_result} STREQUAL "0")
    # only update the bindings if they changed (file not touched -> no rebuild is triggered)
    check_and_update(${BINDINGS_C_DECL_FILENAME} ${new_BINDINGS_C_DECL_FILENAME})
    check_and_update(${BINDINGS_FORTRAN_DECL_FILENAME} ${new_BINDINGS_FORTRAN_DECL_FILENAME})
else()
    message(FATAL_ERROR "GENERATING BINDINGS FAILED. Possibly you cross-compiled the bindings generator for a target "
        " which cannot be executed on this host. Consider using the cross-compilation option.\n Exit code: ${generate_result}\n${generate_out}")
endif()
