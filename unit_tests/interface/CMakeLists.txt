# generate a repository to inspect the preprocessor hackery and add it as a custom test (to be able to add a dependency)
add_executable( generated_repository.cpp plain_repository_generator.cpp )
target_compile_options( generated_repository.cpp BEFORE PUBLIC "-E")
set_target_properties( generated_repository.cpp PROPERTIES RULE_LAUNCH_LINK "${CMAKE_CURRENT_SOURCE_DIR}/parse_preprocessor_output.sh")

set( GENERATED_REPOSITORY "${CMAKE_CURRENT_BINARY_DIR}/generated_repository.cpp" )
configure_file( custom_test_generated_repository.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/custom_test_generated_repository.cpp )

add_executable( custom_test_generated_repository custom_test_generated_repository.cpp )
target_link_libraries(custom_test_generated_repository ${exe_LIBS} gtest_main)
add_dependencies( custom_test_generated_repository generated_repository.cpp )
gridtools_add_test(custom_test_generated_repository ${TEST_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository)

# collect test cases
fetch_host_tests(.)
fetch_gpu_tests(.)
