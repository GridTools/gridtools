# The following macro is a trick to generate a file containing pre-processed compiler output (-E flag) 
macro(generate_preprocessed_file preprocessed_file_out cpp_file)
    # check if ${cpp_file} is a file that exists (to protect that the user passes in multiple files or something else)
    if( EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${cpp_file}" )
        # pretend to generate an object file ...
        add_library( ${cpp_file}_OBJECTS OBJECT EXCLUDE_FROM_ALL ${cpp_file} )
        # ... but actually generate the preprocessed file 
        target_compile_options( ${cpp_file}_OBJECTS BEFORE PUBLIC "-E")
    
        # now use the preprocessed file to generate a library (which we will never use) ...
        add_library( dummy_${cpp_file} STATIC EXCLUDE_FROM_ALL $<TARGET_OBJECTS:${cpp_file}_OBJECTS>)
        # ... but before linking we copy the preprocessed file to the given file "preprocessed_file_out"
        # this trick is necessary because $<TARGET_OBJECTS:XXX> is only defined within add_library
        add_custom_command( TARGET dummy_${cpp_file} PRE_LINK
            COMMAND ${CMAKE_COMMAND} -E copy '$<TARGET_OBJECTS:${cpp_file}_OBJECTS>' ${preprocessed_file_out}
        )
        
        # add a target for the generated file
        add_custom_command( OUTPUT ${preprocessed_file_out} COMMAND "" DEPENDS dummy_${cpp_file})
    else()
        message( ERROR "In macro generate_preprocessed_file: ${cpp_file} is not a file" )
    endif()
endmacro(generate_preprocessed_file)

# only CMake > 3.8 because of $<TARGET_OBJECTS:${cpp_file}_OBJECTS> not supported
IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_GREATER 3.8)
    # generate a file where we can inspect the generated repository
    set( GENERATED_REPOSITORY ${CMAKE_CURRENT_BINARY_DIR}/generated_repository.cpp )
    generate_preprocessed_file( ${GENERATED_REPOSITORY} plain_repository_generator.cpp )
    
    # clean the preprocessed file from comments and from everything before our class starts
    add_custom_target( process_generated_repository
      COMMAND sed -i "/^#/d" ${GENERATED_REPOSITORY} # remove all lines starting with '#'
      # remove everything before "class my_repository"
      COMMAND awk "/class my_repository/,0" ${GENERATED_REPOSITORY} > ${GENERATED_REPOSITORY}_tmp
      COMMAND cp ${GENERATED_REPOSITORY}_tmp ${GENERATED_REPOSITORY}
      DEPENDS ${GENERATED_REPOSITORY}
      )
    
    # format the generated file if we can 
    if( CLANG_FORMAT_FOUND )
      add_custom_target( format_generated_repository COMMAND ${CLANG_FORMAT_BIN} -i ${GENERATED_REPOSITORY}
          DEPENDS process_generated_repository
      )
    else()
      #dummy target for dependency
      add_custom_target( format_generated_repository COMMAND "" DEPENDS process_generated_repository) 
    endif()
    
    # generate the driver
    configure_file( custom_test_generated_repository.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository.cpp )
    
    add_executable( custom_test_generated_repository ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository.cpp )
    target_link_libraries(custom_test_generated_repository ${exe_LIBS} gtest_main)
    gridtools_add_test(custom_test_generated_repository ${TEST_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository)
    add_dependencies( custom_test_generated_repository format_generated_repository )
else()
    message("Not building the test for the generated repository (CMAKE should be > 3.8)")
endif()

# collect test cases
fetch_host_tests(.)
fetch_gpu_tests(.)

