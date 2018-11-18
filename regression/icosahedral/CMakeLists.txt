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

if(GT_ENABLE_TARGET_X86)
    foreach(srcfile IN LISTS SOURCES)
        add_executable(${srcfile}_x86_naive ${srcfile}.cpp)
        target_link_libraries(${srcfile}_x86_naive regression_fixture GridToolsTestX86)
        target_compile_definitions(${srcfile}_x86_naive PRIVATE BACKEND_STRATEGY_NAIVE)

        add_executable(${srcfile}_x86_block ${srcfile}.cpp)
        target_link_libraries(${srcfile}_x86_block regression_fixture GridToolsTestX86)

        gridtools_add_test(
            NAME tests.${srcfile}_x86_block_12_33_61
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_x86_block> 12 33 61
            LABELS regression_x86 target_x86
            ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
            )
        gridtools_add_test(
            NAME tests.${srcfile}_x86_block_23_11_43
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_x86_block> 23 11 43
            LABELS regression_x86 target_x86
            ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
            )
        gridtools_add_test(
            NAME tests.${srcfile}_x86_naive_12_33_21
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_x86_naive> 12 33 21
            LABELS regression_x86 target_x86
            ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
            )
        gridtools_add_test(
            NAME tests.${srcfile}_x86_naive_23_11_23
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${srcfile}_x86_naive> 23 11 23
            LABELS regression_x86 target_x86
            ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
            )

        if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_x86_block)
        endif()
    endforeach(srcfile)
endif(GT_ENABLE_TARGET_X86)

if(GT_ENABLE_TARGET_CUDA)
    set(CUDA_SEPARABLE_COMPILATION OFF)
    foreach(srcfile IN LISTS SOURCES)
        add_executable( ${srcfile}_cuda ${srcfile}.cu  )
        target_link_libraries(${srcfile}_cuda regression_fixture GridToolsTestCUDA)

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
endif(GT_ENABLE_TARGET_CUDA)
