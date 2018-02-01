# generate a repository to inspect the preprocessor hackery and add it as a custom test (to be able to add a dependency)
add_executable( generated_repository.cpp plain_repository_generator.cpp )
# preprocess the file:
target_compile_options( generated_repository.cpp BEFORE PUBLIC "-E")
# the script hooks into the link command with a custom script (I couldn't find a clean way to disable the linking step):
set_target_properties( generated_repository.cpp PROPERTIES RULE_LAUNCH_LINK "${CMAKE_CURRENT_SOURCE_DIR}/omit_linking.sh")

set( GENERATED_REPOSITORY "${CMAKE_CURRENT_BINARY_DIR}/generated_repository.cpp" )

add_custom_target( process_generated_repository
  COMMAND sed -i "/^#/d" ${GENERATED_REPOSITORY} # remove all lines starting with '#'
  COMMAND awk -i inplace "/class my_repository/,0" ${GENERATED_REPOSITORY} # remove everything before "class my_repository"
  )
add_dependencies( process_generated_repository generated_repository.cpp )

if( CLANG_FORMAT_FOUND )
  #format the generated_repository.cpp
  add_custom_target( format_generated_repository COMMAND ${CLANG_FORMAT_BIN} -i ${GENERATED_REPOSITORY} )
else()
  add_custom_target( format_generated_repository COMMAND "" ) #dummy target for dependency
endif()
add_dependencies( format_generated_repository process_generated_repository )

# generate the driver
configure_file( custom_test_generated_repository.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository.cpp )

add_executable( custom_test_generated_repository ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository.cpp )
target_link_libraries(custom_test_generated_repository ${exe_LIBS} gtest_main)
gridtools_add_test(custom_test_generated_repository ${TEST_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository)
add_dependencies( custom_test_generated_repository format_generated_repository )


# collect test cases
fetch_host_tests(.)
fetch_gpu_tests(.)
