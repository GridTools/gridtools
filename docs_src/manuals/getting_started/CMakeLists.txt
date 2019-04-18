if (NOT GT_TESTS_ICOSAHEDRAL_GRID)
    function(add_getting_started_test)
        set(options)
        set(one_value_args SOURCE SUFFIX)
        cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

        get_filename_component(_name ${ARG_SOURCE} NAME_WE)
        set(test_name getting_started_${_name}${ARG_SUFFIX})

        add_executable(${test_name} ${ARG_SOURCE})
        target_link_libraries(${test_name} GridToolsTest)

        gridtools_add_test(
            NAME ${test_name}
            COMMAND $<TARGET_FILE:${test_name}>
            LABELS getting_started
            )
    endfunction()

    file(GLOB getting_started_sources "code/test_*.cpp")
    foreach(test_source IN LISTS getting_started_sources)
        add_getting_started_test(SOURCE ${test_source} SUFFIX "")
    endforeach()

    if(GT_ENABLE_BACKEND_CUDA)
        file(GLOB getting_started_sources "code/test_*.cu")
        foreach(test_source IN LISTS getting_started_sources)
            add_getting_started_test(SOURCE ${test_source} SUFFIX "_cuda")
        endforeach()
    endif()

    # Test the CMake example file
    # The generated file should be part of the repository to be able to build the docs without having to run CMake.
    configure_file(code/CMakeLists.txt.in ${CMAKE_CURRENT_LIST_DIR}/code/CMakeLists.txt @ONLY)

    include(ExternalProject)
    ExternalProject_Add(getting_started_cmake_test
        DOWNLOAD_COMMAND ""
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/code
        CMAKE_ARGS
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DGridTools_DIR=${CMAKE_BINARY_DIR} # relies on the GridToolsConfig.cmake in the build directory (the one used for export(package))
        INSTALL_COMMAND ""
    )
endif()
