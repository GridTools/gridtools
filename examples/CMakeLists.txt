## # add_executable(interface1_notemps_naive   interface1_notemps.cpp ${SOURCES} )
## # add_executable(interface1_notemps_block   interface1_notemps.cpp ${SOURCES} )
## # set_target_properties(interface1_notemps_block
## #         PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")

#add_executable(interface2   interface2.cpp ${SOURCES} )

add_executable(interface1_naive   interface1.cpp )
add_executable(interface1_block   interface1.cpp )
set_target_properties(interface1_block
        PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
target_link_libraries(interface1_naive ${exe_LIBS})
target_link_libraries(interface1_block ${exe_LIBS})

add_executable(boundary-condition_naive   boundary-condition.cpp )
add_executable(boundary-condition_block   boundary-condition.cpp )
set_target_properties(boundary-condition_block
        PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")

add_executable(laplacian_naive   laplacian.cpp )
add_executable(laplacian_block   laplacian.cpp )
set_target_properties(laplacian_block
        PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
target_link_libraries(laplacian_naive ${exe_LIBS})
target_link_libraries(laplacian_block ${exe_LIBS})


add_executable(test_interface   expressions.cpp )
target_link_libraries(test_interface ${exe_LIBS})


add_executable(copy_naive   copy_stencil.cpp )
add_executable(copy_block   copy_stencil.cpp )
set_target_properties(copy_block
        PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
target_link_libraries(copy_naive ${exe_LIBS})
target_link_libraries(copy_block ${exe_LIBS})

add_executable(tridiagonal_naive   tridiagonal.cpp )
add_executable(tridiagonal_block   tridiagonal.cpp )
set_target_properties(tridiagonal_block
        PROPERTIES COMPILE_DEFINITIONS "BACKEND_BLOCK")
target_link_libraries(tridiagonal_naive ${exe_LIBS})
target_link_libraries(tridiagonal_block ${exe_LIBS})

add_executable(basic_laplacian basic_laplacian.cpp)
target_link_libraries(basic_laplacian ${exe_LIBS})

add_executable(vanilla_laplacian vanilla_laplacian.cpp)
target_link_libraries(vanilla_laplacian ${exe_LIBS})

if(CUDA_FOUND)
  if(COMPILE_TO_PTX)
#    CUDA_COMPILE_PTX(copy_ptx  copy_stencil.cu)
#    CUDA_COMPILE_PTX(interface1_ptx   interface1.cu)
#    cuda_add_executable( interface1_cuda_ptx ${interface1_ptx})
#    cuda_add_executable( copy_cuda_ptx ${copy_ptx})
 
    cuda_wrap_srcs(copy_cuda PTX generated_files copy_stencil.cu ${SOURCES} )
    cuda_add_executable( copy_cuda copy_stencil.cu )
#    add_custom_target(copy_cuda ${copy_cuda} echo  ${copy_cuda} )
  else(COMPILE_TO_PTX)

    cuda_add_executable( interface1_cuda interface1.cu )
    cuda_add_executable( laplacian_cuda laplacian.cu )
    cuda_add_executable(boundary-condition_cuda   boundary-condition.cu )
    cuda_add_executable( copy_cuda copy_stencil.cu )
    cuda_add_executable( tridiagonal_cuda tridiagonal.cu )
    target_link_libraries(interface1_cuda ${exe_LIBS})
    target_link_libraries(laplacian_cuda ${exe_LIBS})
    target_link_libraries(boundary-condition_cuda ${exe_LIBS})
    target_link_libraries(tridiagonal_cuda ${exe_LIBS})
    target_link_libraries(copy_cuda ${exe_LIBS} )

  endif(COMPILE_TO_PTX)
endif()
endif()

add_subdirectory( communication )
