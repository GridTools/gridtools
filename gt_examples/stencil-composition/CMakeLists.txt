if( STRUCTURED_GRIDS )
    set( EXAMPLES_SRCFILES )
    list( APPEND EXAMPLES_SRCFILES copy_stencil horizontal_diffusion_limited)

      if(ENABLE_X86)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
          add_executable(${srcfile}_x86_naive_ex   ${srcfile}.cpp ${INC_ALL})
          add_executable(${srcfile}_x86_block_ex   ${srcfile}.cpp ${INC_ALL})
          target_link_libraries(${srcfile}_x86_naive_ex ${exe_LIBS})
          target_link_libraries(${srcfile}_x86_block_ex ${exe_LIBS})
          target_compile_definitions(${srcfile}_x86_naive_ex PUBLIC ${X86_BACKEND_DEFINE} "BACKEND_STRATEGY_NAIVE")
          target_compile_definitions(${srcfile}_x86_block_ex PUBLIC ${X86_BACKEND_DEFINE})
          if ( INSTALL_GT_EXAMPLES )
              install(TARGETS
                      ${srcfile}_x86_naive_ex
                      ${srcfile}_x86_block_ex
                      DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/bin" )
          endif()
        endforeach()
    endif(ENABLE_X86)

    if(ENABLE_MC)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES)
          add_executable(${srcfile}_mc_ex   ${srcfile}.cpp ${INC_ALL})
          target_link_libraries(${srcfile}_mc_ex ${exe_LIBS})
          target_compile_definitions(${srcfile}_mc_ex PUBLIC ${MC_BACKEND_DEFINE})

          if ( INSTALL_GT_EXAMPLES )
              install(TARGETS
                      ${srcfile}_mc_ex
                      DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/bin" )
          endif()
        endforeach()
    endif(ENABLE_MC)

    if(ENABLE_CUDA)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
           cuda_add_executable( ${srcfile}_cuda_ex ${srcfile}.cu ${GTEST_FILES} OPTIONS ${GPU_SPECIFIC_FLAGS} "-D${CUDA_BACKEND_DEFINE}")
           target_link_libraries(${srcfile}_cuda_ex ${exe_LIBS})

          if ( INSTALL_GT_EXAMPLES )
              install(TARGETS
                      ${srcfile}_cuda_ex
                      DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/bin" )
          endif()
            endforeach()
      endif(ENABLE_CUDA)
endif()
