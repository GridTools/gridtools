if( STRUCTURED_GRIDS )
    set( EXAMPLES_SRCFILES copy_stencil horizontal_diffusion_limited)

    if(GT_ENABLE_TARGET_X86)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
          add_executable(${srcfile}_x86_naive_ex   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_x86_naive_ex GridToolsTestX86)
          target_compile_definitions(${srcfile}_x86_naive_ex PRIVATE BACKEND_STRATEGY_NAIVE)

          add_executable(${srcfile}_x86_block_ex   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_x86_block_ex GridToolsTestX86)

          if ( GT_INSTALL_EXAMPLES )
              install(TARGETS
                      ${srcfile}_x86_naive_ex
                      ${srcfile}_x86_block_ex
                      DESTINATION "${GT_INSTALL_EXAMPLES_PATH}/examples/bin" )
          endif()
        endforeach()
    endif(GT_ENABLE_TARGET_X86)

    if(ENABLE_MC)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES)
          add_executable(${srcfile}_mc_ex   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_mc_ex GridToolsTestMC )

          if ( GT_INSTALL_EXAMPLES )
              install(TARGETS
                      ${srcfile}_mc_ex
                      DESTINATION "${GT_INSTALL_EXAMPLES_PATH}/examples/bin" )
          endif()
        endforeach()
    endif(ENABLE_MC)

    if(GT_ENABLE_TARGET_CUDA)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
           add_executable( ${srcfile}_cuda_ex ${srcfile}.cu )
           target_link_libraries(${srcfile}_cuda_ex GridToolsTestCUDA )

           if ( GT_INSTALL_EXAMPLES )
              install(TARGETS
                      ${srcfile}_cuda_ex
                      DESTINATION "${GT_INSTALL_EXAMPLES_PATH}/examples/bin" )
          endif()
            endforeach()
        endif(GT_ENABLE_TARGET_CUDA)
endif()
