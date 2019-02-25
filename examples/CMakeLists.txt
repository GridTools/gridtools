# This file is part of the GridTools source tree to drive installation of the examples.

if(GT_INSTALL_EXAMPLES)
    configure_file(CMakeLists.txt.in CMakeLists.txt @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt DESTINATION ${GT_INSTALL_EXAMPLES_PATH})

    if(COMPONENT_STENCIL_COMPOSITION)
        configure_file(stencil-composition/CMakeLists.txt.in ${CMAKE_CURRENT_LIST_DIR}/stencil-composition/CMakeLists.txt @ONLY)
        install(
            DIRECTORY stencil-composition
            DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
            PATTERN "CMakeLists.txt.in" EXCLUDE
            )
    endif()
    
    if(COMPONENT_BOUNDARY_CONDITIONS)
        configure_file(boundary-conditions/CMakeLists.txt.in ${CMAKE_CURRENT_LIST_DIR}/boundary-conditions/CMakeLists.txt @ONLY)
        install(
            DIRECTORY boundary-conditions
            DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
            PATTERN "CMakeLists.txt.in" EXCLUDE)
    endif()
endif()
