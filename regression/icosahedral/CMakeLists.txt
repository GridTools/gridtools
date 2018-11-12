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
    stencil_on_edges_of_cells
    stencil_fused
    stencil_on_neighedge_of_cells
    stencil_on_vertices
    curl
    div
    lap
    )

add_library(unstructured_grid Options.cpp unstructured_grid.cpp)
target_link_libraries(unstructured_grid GridTools)

add_definitions("-DBENCHMARK") # TODO remove after regression refactoring

# special target for executables which are used from performance benchmarks
add_custom_target(perftests)

if(GT_ENABLE_TARGET_X86)
    foreach(srcfile IN LISTS SOURCES)
        add_executable(${srcfile}_x86_naive ${srcfile}.cpp)
        target_link_libraries(${srcfile}_x86_naive gcl gtest_main unstructured_grid GridToolsTestX86)
        target_compile_definitions(${srcfile}_x86_naive PRIVATE BACKEND_STRATEGY_NAIVE)

        add_executable(${srcfile}_x86_block ${srcfile}.cpp)
        target_link_libraries(${srcfile}_x86_block gcl gtest_main unstructured_grid GridToolsTestX86)

        gridtools_add_test( tests.${srcfile}_x86_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_x86_block" "12 33 61")
        gridtools_add_test( tests.${srcfile}_x86_block ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_x86_block" "23 11 43")
        gridtools_add_test( tests.${srcfile}_x86_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_x86_naive" "12 33 21")
        gridtools_add_test( tests.${srcfile}_x86_naive ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_x86_naive" "23 11 23")

        if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_x86_block)
        endif()
    endforeach(srcfile)
endif(GT_ENABLE_TARGET_X86)

if(GT_ENABLE_TARGET_CUDA)
    set(CUDA_SEPARABLE_COMPILATION OFF)
    foreach(srcfile IN LISTS SOURCES)
        add_executable( ${srcfile}_cuda ${srcfile}.cu  )
        target_link_libraries(${srcfile}_cuda gcl gtest_main unstructured_grid GridToolsTestCUDA)

        if( ${srcfile} STREQUAL "stencil_on_vertexes")
            # K length larger than 48 fails for stencil_on_vertexes_cuda... this need to be investigated
            gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_cuda" "77 48 48")
            gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_cuda" "63 55 43")
        elseif(${srcfile} STREQUAL "expandable_parameters_icosahedral" OR ${srcfile} STREQUAL "copy_stencil_icosahedral")
            gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_cuda" "32 48 61")
            gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_cuda" "64 55 43")
        else()
            gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_cuda" "77 48 61")
            gridtools_add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/regression/icosahedral/${srcfile}_cuda" "63 55 43")
        endif()
        if (srcfile IN_LIST SOURCES_PERFTEST)
            add_dependencies(perftests ${srcfile}_cuda)
        endif()
    endforeach(srcfile)
endif(GT_ENABLE_TARGET_CUDA)
