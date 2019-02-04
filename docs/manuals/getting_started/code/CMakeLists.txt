function(add_getting_started_test test_source suffix)
    get_filename_component(_name ${test_source} NAME_WE)
    set(test_name getting_started_${_name}${suffix})
    
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} GridToolsTest)
    
    gridtools_add_test(
        NAME ${test_name}
        SCRIPT ${TEST_SCRIPT}
        COMMAND $<TARGET_FILE:${test_name}>
        LABELS getting_started
        ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
        )
endfunction()

file(GLOB getting_started_sources "test_*.cpp")
foreach(test_source IN LISTS getting_started_sources)
    add_getting_started_test(${test_source} "")
endforeach()

if(GT_ENABLE_TARGET_CUDA)
    file(GLOB getting_started_sources "test_*.cu")
    foreach(test_source IN LISTS getting_started_sources)
        add_getting_started_test(${test_source} "_cuda")
    endforeach()
endif()
