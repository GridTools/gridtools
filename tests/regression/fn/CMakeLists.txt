function(add_fn_testees prefix)
    foreach(backend IN LISTS ARGN)
        set(tgt ${prefix}_${backend})
        add_library(${tgt} INTERFACE)
        target_link_libraries(${tgt} INTERFACE fn_${backend})
        string(TOUPPER ${backend} u_backend)
        target_compile_definitions(${tgt} INTERFACE GT_FN_${u_backend})
        if (backend STREQUAL gpu)
            target_link_libraries(${tgt} INTERFACE storage_gpu)
        elseif (backend STREQUAL naive)
            target_link_libraries(${tgt} INTERFACE storage_cpu_kfirst)
        endif()
    endforeach()
endfunction()
add_fn_testees(fn_testee ${GT_FN_BACKENDS})

function(gridtools_add_fn_regression_test tgt_name)
    gridtools_add_regression_test(${tgt_name} ${ARGN}
        LIB_PREFIX fn_testee
        KEYS ${GT_FN_BACKENDS}
        LABELS fn)
endfunction()

gridtools_add_fn_regression_test(fn_copy SOURCES fn_copy.cpp)
gridtools_add_fn_regression_test(fn_unstructured_nabla SOURCES fn_unstructured_nabla.cpp)
gridtools_add_fn_regression_test(fn_tridiagonal_solve SOURCES fn_tridiagonal_solve.cpp)

if (CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    add_executable(fn_lap fn_lap.cpp)
    target_link_libraries(fn_lap gtest_main gmock gridtools)
    add_test(NAME fn_lap COMMAND $<TARGET_FILE:fn_lap>)

    add_executable(fn_hd fn_hd.cpp)
    target_link_libraries(fn_hd gtest_main gmock gridtools)
    add_test(NAME fn_hd COMMAND $<TARGET_FILE:fn_hd>)

    add_executable(fn_tridiag fn_tridiag.cpp)
    target_link_libraries(fn_tridiag gtest_main gmock gridtools)
    add_test(NAME fn_tridiag COMMAND $<TARGET_FILE:fn_tridiag>)

    add_executable(fn_nabla fn_nabla.cpp)
    target_link_libraries(fn_nabla gtest_main gmock gridtools)
    add_test(NAME fn_nabla COMMAND $<TARGET_FILE:fn_nabla>)
endif()
