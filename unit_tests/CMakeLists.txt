# collect tests
fetch_host_tests(.)
fetch_gpu_tests(.)

if ( USE_GPU )
    if ( STRUCTURED_GRIDS )
        # The work to get icosahedral grids seems a little too much
        # since the grid has dependencies on backends and several
        # includes need to be specified. For now just do structured
        cuda_add_executable( linking_issue linking_issue.cu linking_issue.cpp OPTIONS ${GPU_SPECIFIC_FLAGS})
        set_target_properties( linking_issue PROPERTIES COMPILE_FLAGS ${GPU_SPECIFIC_FLAGS})
        target_link_libraries( linking_issue ${exe_LIBS} gtest_main)
        gridtools_add_test(linking_issue ${TEST_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/linking_issue)
    endif()
endif()

# include all subdirs
add_subdirectory( common )
add_subdirectory( communication )
add_subdirectory( boundary-conditions )
add_subdirectory( stencil-composition )
add_subdirectory( storage )
