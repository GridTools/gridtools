# generate a repository to inspect the preprocessor hackery and add it as a custom test (to be able to add a dependency)
add_custom_command(OUTPUT generated_repository.cpp
    COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} -DGT_PARSE_PREPROCESSOR -DBOOST_PP_VARIADICS=1 -E ${CMAKE_CURRENT_SOURCE_DIR}/plain_repository_generator.cpp -I${CMAKE_SOURCE_DIR}/include > ${CMAKE_CURRENT_SOURCE_DIR}/generated_repository_tmp.cpp
    COMMAND sed -i "/^#/d" ${CMAKE_CURRENT_SOURCE_DIR}/generated_repository_tmp.cpp
    COMMAND awk "/class my_repository/,0" ${CMAKE_CURRENT_SOURCE_DIR}/generated_repository_tmp.cpp | clang-format > ${CMAKE_CURRENT_SOURCE_DIR}/generated_repository.cpp
    COMMAND rm ${CMAKE_CURRENT_SOURCE_DIR}/generated_repository_tmp.cpp 
    )
add_custom_target( generate_repo DEPENDS generated_repository.cpp )
add_executable( custom_test_generated_repository custom_test_generated_repository.cpp )
target_link_libraries(custom_test_generated_repository ${exe_LIBS} gtest_main)
add_dependencies( custom_test_generated_repository generate_repo )
gridtools_add_test(custom_test_generated_repository ${TEST_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository)

# collect test cases
fetch_host_tests(.)
fetch_gpu_tests(.)
