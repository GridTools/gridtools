add_executable(fn_lap fn_lap.cpp)
target_link_libraries(fn_lap gtest_main gmock gridtools)
add_test(NAME fn_lap COMMAND $<TARGET_FILE:fn_lap>)

add_executable(fn_nabla fn_nabla.cpp)
target_link_libraries(fn_nabla gtest_main gmock gridtools)
add_test(NAME fn_nabla COMMAND $<TARGET_FILE:fn_nabla>)
