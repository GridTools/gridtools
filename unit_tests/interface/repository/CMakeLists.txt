# The following macro is a trick to generate a file containing pre-processed compiler output (-E flag)
macro(generate_preprocessed_file preprocessed_file_out cpp_file)
    # check if ${cpp_file} is a file that exists (to protect that the user passes in multiple files or something else)
    if( EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${cpp_file}" )
        # pretend to generate an object file ...
        add_library( ${cpp_file}_OBJECTS OBJECT EXCLUDE_FROM_ALL ${cpp_file} )
        # ... but actually generate the preprocessed file
        target_compile_options( ${cpp_file}_OBJECTS BEFORE PUBLIC -E ${GT_CXX_FLAGS})
        target_link_libraries( ${cpp_file}_OBJECTS common )

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

if (GT_ENABLE_BACKEND_X86)
    # generate a file where we can inspect the generated repository
    set( GENERATED_REPOSITORY ${CMAKE_CURRENT_BINARY_DIR}/generated_repository.cpp )
    generate_preprocessed_file( ${GENERATED_REPOSITORY} plain_repository_generator.cpp )

    # clean the preprocessed file from comments and from everything before our class starts
    add_custom_command(OUTPUT ${GENERATED_REPOSITORY}
      # remove all lines starting with '#'
      # remove everything before "class my_repository"
      COMMAND sed "/^#/d" ${GENERATED_REPOSITORY} | awk "/class my_repository/,0" > ${GENERATED_REPOSITORY}_tmp
      COMMAND cp ${GENERATED_REPOSITORY}_tmp ${GENERATED_REPOSITORY}
      APPEND
      )

    # format the generated file if we can
    if( CLANG_FORMAT_FOUND )
        add_custom_command(OUTPUT ${GENERATED_REPOSITORY}
            COMMAND ${CLANG_FORMAT_BIN} -i ${GENERATED_REPOSITORY}
            APPEND
      )
    endif()

    # generate the driver
    configure_file( custom_test_generated_repository.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository.cpp )

    add_executable( custom_test_generated_repository ${CMAKE_CURRENT_BINARY_DIR}/custom_test_generated_repository.cpp )
    target_link_libraries(custom_test_generated_repository gtest_main GridToolsTestX86 )
    gridtools_add_test(
        NAME custom_test_generated_repository
        SCRIPT ${TEST_SCRIPT}
        COMMAND $<TARGET_FILE:custom_test_generated_repository>
        LABELS unittest_x86 target_x86
        ENVIRONMENT ${TEST_HOST_ENVIRONMENT}
        )
    add_dependencies(custom_test_generated_repository generated_repository)
endif()

# collect test cases
fetch_x86_tests(. LABELS unittest_x86 )
fetch_gpu_tests(. LABELS unittest_cuda )
fetch_mc_tests(. LABELS unittest_mc )

if(GT_SINGLE_PRECISION)
    set(prec float)
else()
    set(prec double)
endif()

gt_add_bindings_library(repository_${prec} SOURCES exported_repository.cpp FORTRAN_MODULE_NAME repository)
target_link_libraries(repository_${prec} PUBLIC gridtools gtest)
if(GT_SINGLE_PRECISION)
    target_compile_definitions(repository_${prec} PUBLIC GT_FLOAT_PRECISION=4)
else()
    target_compile_definitions(repository_${prec} PUBLIC GT_FLOAT_PRECISION=8)
endif()

if (CMAKE_Fortran_COMPILER_LOADED)
    include(fortran_helpers)

    add_library(repository_fortran_lib test_repository.f90)
    target_link_libraries(repository_fortran_lib repository_${prec}_fortran)
    gt_enable_fortran_preprocessing_on_target(repository_fortran_lib)

    if (GT_ENABLE_BACKEND_X86)
        add_custom_x86_test(
            TARGET custom_test_exported_repository
            SOURCES custom_test_exported_repository.cpp
            LABELS unittest_x86
            )
        target_link_libraries(custom_test_exported_repository_x86
            repository_fortran_lib
            )
    endif()
    # TODO Fix this test. This test does not work right now because repository_${proc}_fortran
    # is not compiled with ACC. ACC cannot be activated for CUDA 8.X because it requires
    # gfortran <= 5.X
    # if (GT_ENABLE_BACKEND_CUDA)
        # add_custom_gpu_test(
            # TARGET custom_test_exported_repository
            # SOURCES custom_test_exported_repository.cpp
            # LABELS unittest_cuda
            # )
        # target_link_libraries(custom_test_exported_repository_cuda
            # repository_fortran_lib
            # )
    # endif()
    if (GT_ENABLE_BACKEND_MC AND NOT GT_ENABLE_BACKEND_X86)
        add_custom_mc_test(
            TARGET custom_test_exported_repository
            SOURCES custom_test_exported_repository.cpp
            LABELS unittest_mc
            )
        target_link_libraries(custom_test_exported_repository_mc
            repository_fortran_lib
            )
    endif()


endif()
