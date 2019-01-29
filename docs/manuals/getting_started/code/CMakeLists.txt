function(add_getting_started_test test_source variant)
    get_filename_component(_name ${test_source} NAME_WE)
    set(test_name getting_started_${_name})
    if(variant)
        set(test_name ${test_name}_${variant})
    endif()    
    string(TOLOWER ${test_name} test_name)
    
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} GridToolsTest)
    if(variant)
        target_compile_definitions(${test_name} PRIVATE ${variant})
    endif()
    
    gridtools_add_test(
        NAME ${test_name}
        SCRIPT ${TEST_SCRIPT}
        COMMAND $<TARGET_FILE:${test_name}>
        LABELS getting_started
        ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
        )
endfunction()

# all .cpp files are compiled
file(GLOB getting_started_sources "*.cpp")
foreach(test_source IN LISTS getting_started_sources)
    add_getting_started_test(${test_source} "")
endforeach()

# add some variants
add_getting_started_test("gt_smoothing.cpp" "VARIANT1")
add_getting_started_test("gt_smoothing.cpp" "VARIANT2")
add_getting_started_test("gt_smoothing.cpp" "VARIANT3")

if(GT_ENABLE_TARGET_CUDA)
    file(GLOB getting_started_sources "*.cu")
    foreach(test_source IN LISTS getting_started_sources)
        add_getting_started_test(${test_source} "")
    endforeach()
    
    add_getting_started_test("gt_smoothing_cuda.cu" "VARIANT1")
    add_getting_started_test("gt_smoothing_cuda.cu" "VARIANT2")
    add_getting_started_test("gt_smoothing_cuda.cu" "VARIANT3")
endif()
