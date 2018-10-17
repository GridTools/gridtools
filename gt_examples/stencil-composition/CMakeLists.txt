if( STRUCTURED_GRIDS )
    set( EXAMPLES_SRCFILES )
    list( APPEND EXAMPLES_SRCFILES copy_stencil horizontal_diffusion_limited)

      if(ENABLE_HOST)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
          add_executable(${srcfile}_host_naive_ex   ${srcfile}.cpp ${INC_ALL})
          add_executable(${srcfile}_host_block_ex   ${srcfile}.cpp ${INC_ALL})
          target_link_libraries(${srcfile}_host_naive_ex ${exe_LIBS})
          target_link_libraries(${srcfile}_host_block_ex ${exe_LIBS})
          target_compile_options(${srcfile}_host_naive_ex PUBLIC ${GT_CXX_FLAGS} -D${HOST_BACKEND_DEFINE} -DBACKEND_STRATEGY_NAIVE)
          target_compile_options(${srcfile}_host_block_ex PUBLIC ${GT_CXX_FLAGS} -D${HOST_BACKEND_DEFINE})
          target_include_directories(${srcfile}_host_naive_ex
               PRIVATE
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
          )
          target_include_directories(${srcfile}_host_block_ex
               PRIVATE
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
          )

          if ( INSTALL_GT_EXAMPLES )
              install(TARGETS
                      ${srcfile}_host_naive_ex
                      ${srcfile}_host_block_ex
                      DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/bin" )
          endif()
        endforeach()
    endif(ENABLE_HOST)

    if(ENABLE_MIC)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES)
          add_executable(${srcfile}_mic_ex   ${srcfile}.cpp ${INC_ALL})
          target_link_libraries(${srcfile}_mic_ex ${exe_LIBS})
          target_compile_options(${srcfile}_mic_ex PUBLIC ${GT_CXX_FLAGS} -D${MIC_BACKEND_DEFINE})
          target_include_directories(${srcfile}_mic_ex
               PRIVATE
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
          )

          if ( INSTALL_GT_EXAMPLES )
              install(TARGETS
                      ${srcfile}_mic_ex
                      DESTINATION "${INSTALL_GT_EXAMPLES_PATH}/examples/bin" )
          endif()
        endforeach()
    endif(ENABLE_MIC)

    if(ENABLE_CUDA)
        foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
           cuda_add_executable( ${srcfile}_cuda_ex ${srcfile}.cu ${GTEST_FILES} OPTIONS ${GPU_SPECIFIC_FLAGS} "-D${CUDA_BACKEND_DEFINE}")
           target_link_libraries(${srcfile}_cuda_ex ${exe_LIBS})
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
      endif(ENABLE_CUDA)
endif()
