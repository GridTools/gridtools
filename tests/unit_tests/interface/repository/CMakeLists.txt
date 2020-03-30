# The following macro is a trick to generate a file containing pre-processed compiler output (-E flag)
macro(generate_preprocessed_file preprocessed_file_out cpp_file)
    # check if ${cpp_file} is a file that exists (to protect that the user passes in multiple files or something else)
    if( EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${cpp_file}" )
        # pretend to generate an object file ...
        add_library( ${cpp_file}_OBJECTS OBJECT EXCLUDE_FROM_ALL ${cpp_file} )
        # ... but actually generate the preprocessed file
        target_compile_options( ${cpp_file}_OBJECTS BEFORE PUBLIC -E ${GT_CXX_FLAGS})
        target_link_libraries( ${cpp_file}_OBJECTS GridToolsTest )
        target_link_libraries( ${cpp_file}_OBJECTS cpp_bindgen_interface )

        # add a target for the generated file
        add_custom_command( OUTPUT ${preprocessed_file_out}
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${cpp_file}_OBJECTS> ${preprocessed_file_out}
            DEPENDS ${cpp_file}_OBJECTS $<TARGET_OBJECTS:${cpp_file}_OBJECTS>
            COMMENT "Extract generated code"
            )

        get_filename_component(generated_target_name ${preprocessed_file_out} NAME_WE)
        add_custom_target(${generated_target_name}
            DEPENDS ${preprocessed_file_out}
            )
    else()
        message( ERROR "In macro generate_preprocessed_file: ${cpp_file} is not a file" )
    endif()
endmacro(generate_preprocessed_file)

# generate a file where we can inspect the generated repository
set(GENERATED_REPOSITORY ${CMAKE_CURRENT_BINARY_DIR}/generated_repository.cpp)
generate_preprocessed_file(${GENERATED_REPOSITORY} plain_repository_generator.cpp)

# clean the preprocessed file from comments and from everything before our class starts
add_custom_command(OUTPUT ${GENERATED_REPOSITORY}
        # remove all lines starting with '#'
        # remove everything before "class my_repository"
        COMMAND sed "/^#/d" ${GENERATED_REPOSITORY} | awk "/namespace my_repository_impl_/,0" > ${GENERATED_REPOSITORY}_tmp
        COMMAND cp ${GENERATED_REPOSITORY}_tmp ${GENERATED_REPOSITORY}
        APPEND)

## clang tools
find_package(ClangTools)

# format the generated file if we can
if(CLANG_FORMAT_FOUND)
    add_custom_command(OUTPUT ${GENERATED_REPOSITORY}
            COMMAND ${CLANG_FORMAT_BIN} -i ${GENERATED_REPOSITORY}
            APPEND)
endif()

# generate the driver
configure_file(custom_test_generated_repository.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository.cpp)

gridtools_add_unit_test(custom_test_generated_repository
        SOURCES ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository.cpp
        LIBRARIES cpp_bindgen_interface
        NO_NVCC)
add_dependencies(custom_test_generated_repository generated_repository)

gridtools_add_unit_test(test_repository SOURCES test_repository.cpp LIBRARIES cpp_bindgen_interface)

if (CMAKE_Fortran_COMPILER_LOADED)
    bindgen_add_library(exported_repository SOURCES exported_repository.cpp FORTRAN_MODULE_NAME exported_repository)
    target_link_libraries(exported_repository PUBLIC gridtools gtest)

    add_library(repository_fortran_lib test_repository.f90)
    target_link_libraries(repository_fortran_lib exported_repository_fortran)

    gridtools_add_unit_test(custom_test_exported_repository
            SOURCES custom_test_exported_repository.cpp
            LIBRARIES repository_fortran_lib
            NO_NVCC)
endif()
