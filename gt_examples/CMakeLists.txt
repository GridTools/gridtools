macro(install_example name)
    if ( GT_INSTALL_EXAMPLES )
        install(TARGETS ${name}
                DESTINATION "${GT_INSTALL_EXAMPLES_PATH}/examples/bin" )
    endif()
endmacro()

if ( COMPONENT_STENCIL_COMPOSITION )
   add_subdirectory( stencil-composition )
endif()

if ( COMPONENT_BOUNDARY_CONDITIONS )
   add_subdirectory( boundary-conditions )
endif()

if( GT_TESTS_STRUCTURED_GRID )
    if ( COMPONENT_C_BINDINGS )
       add_subdirectory( c_bindings )
    endif()
endif()
