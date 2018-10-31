function(check_and_update target_file new_file)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E compare_files ${target_file} ${new_file}
        RESULT_VARIABLE compare_result
        )

    if(${compare_result} EQUAL 0)
        message(STATUS "${target_file} is up-to-date")
        file(REMOVE ${new_file})
    else()
        message(WARNING "${target_file} was generated! "
            "If you ship the generated bindings with your sources, don't forget to ship this updated file (and its variants). "
            "Otherwise, this warning can be ignored.")
        file(RENAME ${new_file} ${target_file})
    endif()
endfunction()

# Generate bindings and compare against existing ones
message(STATUS "Generating bindings for library ${FORTRAN_MODULE_NAME}")

# run generator
execute_process(COMMAND ${GENERATOR} ${BINDINGS_C_DECL_FILENAME}.new ${BINDINGS_FORTRAN_DECL_FILENAME}.new ${FORTRAN_MODULE_NAME})

# only update the bindings if they changed (file not touched -> no rebuild is triggered)
check_and_update(${BINDINGS_C_DECL_FILENAME} ${BINDINGS_C_DECL_FILENAME}.new)
check_and_update(${BINDINGS_FORTRAN_DECL_FILENAME} ${BINDINGS_FORTRAN_DECL_FILENAME}.new)
