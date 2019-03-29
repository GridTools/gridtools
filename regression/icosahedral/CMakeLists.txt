set(SOURCES_PERFTEST
    stencil_on_edges_multiplefields
    stencil_on_cells
    stencil_on_neighcell_of_edges
    stencil_manual_fold
    )
set(SOURCES
    ${SOURCES_PERFTEST}
    copy_stencil_icosahedral
    expandable_parameters_icosahedral
    stencil_on_cells_color
    stencil_on_edges
    stencil_fused
    stencil_on_neighedge_of_cells
    stencil_on_vertices
    curl
    div
    lap
    )

# special target for executables which are used from performance benchmarks
add_custom_target(perftests)

if(GT_ENABLE_BACKEND_X86)
    foreach(srcfile IN LISTS SOURCES)
        add_executable(${srcfile}_x86 ${srcfile}.cpp)
        target_link_libraries(${srcfile}_x86 regression_main GridToolsTestX86)

        gridtools_add_test(
            NAME tests.${srcfile}_x86_12_33_61
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_x86> 12 33 61
            LABELS regression_x86 target_x86
            ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
            )
        gridtools_add_test(
            NAME tests.${srcfile}_x86_23_11_43
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_x86> 23 11 43
            LABELS regression_x86 target_x86
            ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
            )

        if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_x86)
        endif()
    endforeach(srcfile)
endif(GT_ENABLE_BACKEND_X86)

if(GT_ENABLE_BACKEND_NAIVE)
    foreach(srcfile IN LISTS SOURCES)
        add_executable(${srcfile}_naive ${srcfile}.cpp)
        target_link_libraries(${srcfile}_naive regression_main GridToolsTestNAIVE)

        gridtools_add_test(
            NAME tests.${srcfile}_naive_12_33_21
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_naive> 12 33 21
            LABELS regression_naive target_naive
            ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
            )
        gridtools_add_test(
            NAME tests.${srcfile}_naive_23_11_23
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_naive> 23 11 23
            LABELS regression_naive target_naive
            ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
            )

        if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_naive)
        endif()
    endforeach(srcfile)
endif(GT_ENABLE_BACKEND_NAIVE)

if(GT_ENABLE_BACKEND_CUDA)
    set(CUDA_SEPARABLE_COMPILATION OFF)
    foreach(srcfile IN LISTS SOURCES)
        add_executable( ${srcfile}_cuda ${srcfile}.cu  )
        target_link_libraries(${srcfile}_cuda regression_main GridToolsTestCUDA)

        gridtools_add_test(
            NAME tests.${srcfile}_cuda_77_48_61
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_cuda> 77 48 61
            LABELS regression_cuda target_cuda
            ENVIRONMENT ${TEST_CUDA_ENVIRONMENT}
            )
        gridtools_add_test(
            NAME tests.${srcfile}_cuda_63_55_43
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_cuda> 63 55 43
            LABELS regression_cuda target_cuda
            ENVIRONMENT ${TEST_CUDA_ENVIRONMENT}
            )

        if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_cuda)
        endif()
    endforeach(srcfile)
endif(GT_ENABLE_BACKEND_CUDA)
