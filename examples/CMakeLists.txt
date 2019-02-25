# This file is part of the GridTools source tree to drive installation of the examples.

if(GT_INSTALL_EXAMPLES)
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
