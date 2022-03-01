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

    add_executable(fn_nested_reduce fn_nested_reduce.cpp)
    target_link_libraries(fn_nested_reduce gtest_main gmock gridtools)
    add_test(NAME fn_nested_reduce COMMAND $<TARGET_FILE:fn_nested_reduce>)
endif()
