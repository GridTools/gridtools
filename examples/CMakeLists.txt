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

add_executable(basic_laplacian basic_laplacian.cpp)
target_link_libraries(basic_laplacian ${exe_LIBS})

if(CUDA_FOUND)
    cuda_add_executable( interface1_cuda interface1.cu )
    #add_dependencies(  )

    cuda_add_executable( laplacian_cuda laplacian.cu )
    cuda_add_executable(boundary-condition_cuda   boundary-condition.cu )
    target_link_libraries(interface1_cuda ${exe_LIBS})
    target_link_libraries(laplacian_cuda ${exe_LIBS})
    target_link_libraries(boundary-condition_cuda ${exe_LIBS})
endif()

add_subdirectory( communication )
