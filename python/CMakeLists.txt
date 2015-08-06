find_program(PYTHON "python")

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/gridtools/__init__.py")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    if(NOT PYTHON_INSTALL_PREFIX STREQUAL " ") 
      execute_process(COMMAND ${CMAKE_SOURCE_DIR}/python/python_venv.sh ${CMAKE_SOURCE_DIR} ${PYTHON_INSTALL_PREFIX} OUTPUT_FILE ${OUTPUT}  ERROR_VARIABLE DUMMY ) 
      add_custom_target(python ALL COMMAND ${CMAKE_SOURCE_DIR}/python/python_install.sh ${CMAKE_SOURCE_DIR} ${PYTHON_INSTALL_PREFIX} COMMENT "Installing python")
      message("Virtual environment installed in " ${PYTHON_INSTALL_PREFIX})
      gridtools_add_test(python_tests ${TEST_SCRIPT} "${CMAKE_SOURCE_DIR}/python/python_run_tests.sh" "${CMAKE_SOURCE_DIR} ${PYTHON_INSTALL_PREFIX}")
    endif()

endif()
