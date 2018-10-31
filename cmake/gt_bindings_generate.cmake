message(STATUS "Generating bindings for library ${FORTRAN_MODULE_NAME}")

# run generator
execute_process(COMMAND ${GENERATOR} ${BINDINGS_C_DECL_FILENAME}.new ${BINDINGS_FORTRAN_DECL_FILENAME}.new ${FORTRAN_MODULE_NAME})

# check if generated files 
execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files ${BINDINGS_C_DECL_FILENAME} ${BINDINGS_C_DECL_FILENAME}.new
    RESULT_VARIABLE c_decl_uptodate
    )
execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files ${BINDINGS_FORTRAN_DECL_FILENAME} ${BINDINGS_FORTRAN_DECL_FILENAME}.new
    RESULT_VARIABLE fortran_decl_uptodate
    )

if(${c_decl_uptodate} EQUAL 0)
    message(STATUS "${BINDINGS_C_DECL_FILENAME} is up-to-date")
    file(REMOVE ${BINDINGS_C_DECL_FILENAME}.new)
else()
    message(STATUS "${BINDINGS_C_DECL_FILENAME} was generated")
    file(RENAME ${BINDINGS_C_DECL_FILENAME}.new ${BINDINGS_C_DECL_FILENAME})
endif()

if(${fortran_decl_uptodate} EQUAL 0)
    message(STATUS "${BINDINGS_FORTRAN_DECL_FILENAME} is up-to-date")
    file(REMOVE ${BINDINGS_FORTRAN_DECL_FILENAME}.new)
else()
    message(STATUS "${BINDINGS_FORTRAN_DECL_FILENAME} was generated")
    file(RENAME ${BINDINGS_FORTRAN_DECL_FILENAME}.new ${BINDINGS_FORTRAN_DECL_FILENAME})
endif()
