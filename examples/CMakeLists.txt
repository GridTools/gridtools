# This file is part of the GridTools source tree to drive installation of the examples.

if(GT_INSTALL_EXAMPLES)
    configure_file(CMakeLists.txt.in CMakeLists.txt @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt DESTINATION ${GT_INSTALL_EXAMPLES_PATH})

    if(COMPONENT_STENCIL_COMPOSITION)
        configure_file(stencil-composition/CMakeLists.txt.in stencil-composition/CMakeLists.txt @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/stencil-composition/CMakeLists.txt DESTINATION ${GT_INSTALL_EXAMPLES_PATH}/stencil-composition)
        install(
            DIRECTORY stencil-composition
            DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
            PATTERN "CMakeLists.txt.in" EXCLUDE
            )
    endif()
    
    if(COMPONENT_BOUNDARY_CONDITIONS)
        configure_file(boundary-conditions/CMakeLists.txt.in boundary-conditions/CMakeLists.txt @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/boundary-conditions/CMakeLists.txt DESTINATION ${GT_INSTALL_EXAMPLES_PATH}/boundary-conditions)
        install(
            DIRECTORY boundary-conditions
            DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
            PATTERN "CMakeLists.txt.in" EXCLUDE)
    endif()
endif()
