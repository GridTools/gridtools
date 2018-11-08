if( STRUCTURED_GRIDS )
    set( EXAMPLES_SRCFILES )
    list( APPEND EXAMPLES_SRCFILES copy_stencil horizontal_diffusion_limited)

    if(GT_ENABLE_TARGET_X86)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
          add_executable(${srcfile}_x86_naive_ex   ${srcfile}.cpp)
          add_executable(${srcfile}_x86_block_ex   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_x86_naive_ex stencil-composition)
          target_link_libraries(${srcfile}_x86_block_ex stencil-composition)
          target_compile_options(${srcfile}_x86_naive_ex PUBLIC ${GT_CXX_FLAGS} -D${X86_BACKEND_DEFINE} -DBACKEND_STRATEGY_NAIVE)
          target_compile_options(${srcfile}_x86_block_ex PUBLIC ${GT_CXX_FLAGS} -D${X86_BACKEND_DEFINE})
          target_include_directories(${srcfile}_x86_naive_ex
               PRIVATE
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
          )
          target_include_directories(${srcfile}_x86_block_ex
               PRIVATE
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
          )

          if ( INSTALL_GT_EXAMPLES )
              install(TARGETS
                      ${srcfile}_x86_naive_ex
                      ${srcfile}_x86_block_ex
                      DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/bin" )
          endif()
        endforeach()
    endif(GT_ENABLE_TARGET_X86)

    if(ENABLE_MC)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES)
          add_executable(${srcfile}_mc_ex   ${srcfile}.cpp)
          target_link_libraries(${srcfile}_mc_ex stencil-composition )
          target_compile_options(${srcfile}_mc_ex PUBLIC ${GT_CXX_FLAGS} -D${MC_BACKEND_DEFINE})
          target_include_directories(${srcfile}_mc_ex
               PRIVATE
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
          )

          if ( INSTALL_GT_EXAMPLES )
              install(TARGETS
                      ${srcfile}_mc_ex
                      DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/bin" )
          endif()
        endforeach()
    endif(ENABLE_MC)

    if(GT_ENABLE_TARGET_CUDA)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
           add_executable( ${srcfile}_cuda_ex ${srcfile}.cu )
           target_compile_options( ${srcfile}_cuda_ex PUBLIC ${GT_CXX_FLAGS} ${GT_CUDA_FLAGS} ${GPU_SPECIFIC_FLAGS} -D${CUDA_BACKEND_DEFINE})
           target_link_libraries(${srcfile}_cuda_ex stencil-composition )
           target_include_directories(${srcfile}_cuda_ex
               PRIVATE
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
           )

          if ( INSTALL_GT_EXAMPLES )
              install(TARGETS
                      ${srcfile}_cuda_ex
                      DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/bin" )
          endif()
            endforeach()
        endif(GT_ENABLE_TARGET_CUDA)
endif()
