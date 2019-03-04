# This file is part of the GridTools source tree to drive installation of the examples.

function(install_example)
    set(options)
    # - if needed later split into mc and cuda sources (then add extension)
    set(one_value_args DIRECTORY)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ARG "${options}" "${one_value_args};" "${multi_value_args}" ${ARGN})

    string(REPLACE ";" " " EXAMPLES_SRCFILES "${ARG_SOURCES}")
    configure_file(cmake_skeletons/CMakeLists.txt.example.in ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${ARG_DIRECTORY}/CMakeLists.txt @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${ARG_DIRECTORY}/CMakeLists.txt DESTINATION ${GT_INSTALL_EXAMPLES_PATH}/${ARG_DIRECTORY})
    install(
        DIRECTORY ${ARG_DIRECTORY}
        DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
        PATTERN "CMakeLists.txt.in" EXCLUDE
        )

    list(APPEND enabled_examples ${ARG_DIRECTORY})
    set(enabled_examples ${enabled_examples} PARENT_SCOPE)
endfunction()

if(GT_INSTALL_EXAMPLES)
    set(enabled_examples) # used in CMakeLists.txt.in, filled in install_example()
    if(COMPONENT_STENCIL_COMPOSITION)
        install_example(DIRECTORY stencil-composition SOURCES copy_stencil horizontal_diffusion_limited tridiagonal_solver)
    endif()

    if(COMPONENT_BOUNDARY_CONDITIONS)
        install_example(DIRECTORY boundary-conditions SOURCES boundary-conditions boundary-conditions_provided)
    endif()

    configure_file(cmake_skeletons/CMakeLists.txt.driver.in ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeLists.txt @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeLists.txt DESTINATION ${GT_INSTALL_EXAMPLES_PATH})
endif()
