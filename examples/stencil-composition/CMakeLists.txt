set( EXAMPLES_SRCFILES )
list( APPEND EXAMPLES_SRCFILES copy_stencil )

#  if(ENABLE_HOST)
    foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
      add_executable(${srcfile}_host_naive_ex   ${srcfile}.cpp ${INC_ALL})
      add_executable(${srcfile}_host_block_ex   ${srcfile}.cpp ${INC_ALL})
      target_link_libraries(${srcfile}_host_naive_ex ${exe_LIBS})
      target_link_libraries(${srcfile}_host_block_ex ${exe_LIBS})
      target_compile_definitions(${srcfile}_host_naive_ex PUBLIC ${HOST_BACKEND_DEFINE} "BACKEND_STRATEGY_NAIVE")
      target_compile_definitions(${srcfile}_host_block_ex PUBLIC ${HOST_BACKEND_DEFINE})

    endforeach()
# endif(ENABLE_HOST)

  ## if(ENABLE_MIC)
  ##   foreach(srcfile IN LISTS EXAMPLES_SRCFILES)
  ##     add_executable(${srcfile}_mic_ex   ${srcfile}.cpp ${INC_ALL})
  ##     target_link_libraries(${srcfile}_mic_ex ${exe_LIBS})
  ##     target_compile_definitions(${srcfile}_mic_ex PUBLIC ${MIC_BACKEND_DEFINE})

  ##   endforeach()
  ## endif(ENABLE_MIC)

  ## if(ENABLE_CUDA)
  ##       foreach(srcfile IN LISTS EXAMPLES_SRCFILES )
  ##            #set(CUDA_SEPARABLE_COMPILATION ON)
  ##            cuda_add_executable( ${srcfile}_cuda_ex ${srcfile}.cu ${GTEST_FILES} OPTIONS ${GPU_SPECIFIC_FLAGS} "-D${CUDA_BACKEND_DEFINE}")
  ##            target_link_libraries(${srcfile}_cuda_ex ${exe_LIBS})

  ##       endforeach()
  ## endif(ENABLE_CUDA)
