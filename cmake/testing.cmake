enable_testing()

####################################################################################
########################### GET GTEST LIBRARY ############################
####################################################################################

# ===============
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tools/googletest")

if( NOT GT_GCL_ONLY )
    if( GT_USE_MPI )
        add_library( mpi_gtest_main include/gridtools/tools/mpi_unit_test_driver/mpi_test_driver.cpp )
        target_link_libraries(mpi_gtest_main gtest MPI::MPI_CXX GridToolsTest)
        if (GT_ENABLE_TARGET_CUDA)
            target_include_directories( mpi_gtest_main PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
        endif()
    endif()
endif()

####################################################################################
######################### ADDITIONAL TEST MODULE FUNCTIONS #########################
####################################################################################

function(add_fetched_tests_helper test_sources suffix gt_library labels)
    foreach(test_source IN LISTS test_sources )
        # create a nice name for the test case
        get_filename_component (unit_test ${test_source} NAME_WE )
        set(unit_test "${unit_test}_${suffix}")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${unit_test})
        # create the test
        add_executable (${unit_test} ${test_source} )
        target_link_libraries(${unit_test} ${gt_library} c_bindings_generator c_bindings_handle gtest gmock_main)

        add_test (NAME ${unit_test} COMMAND ${exe} )
        set_tests_properties(${unit_test} PROPERTIES LABELS "${labels}")
        gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})
    endforeach()
endfunction()

# This function will fetch all x86 test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_x86_tests subfolder)
    set(options)
    set(one_value_args)
    set(multi_value_args LABELS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(labels ${___LABELS})
    list(APPEND labels target_x86)

    if (GT_ENABLE_TARGET_X86)
        # get all source files in the current directory
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cpp" )
        add_fetched_tests_helper("${test_sources}" x86 GridToolsTestX86 "${labels}")
    endif(GT_ENABLE_TARGET_X86)
endfunction(fetch_x86_tests)

# This function will fetch all mc test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_mc_tests subfolder)
    set(options)
    set(one_value_args)
    set(multi_value_args LABELS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(labels ${___LABELS})
    list(APPEND labels target_mc)

    if (GT_ENABLE_TARGET_MC)
        # get all source files in the current directory
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cpp" )
        add_fetched_tests_helper("${test_sources}" mc GridToolsTestMC "${labels}")
    endif(GT_ENABLE_TARGET_MC)
endfunction(fetch_mc_tests)

# This function will fetch all gpu test cases in the given directory.
# Only used for nvcc compilations
function(fetch_gpu_tests subfolder)
    set(options)
    set(one_value_args)
    set(multi_value_args LABELS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(labels ${___LABELS})
    list(APPEND labels target_cuda)

    if(GT_ENABLE_TARGET_CUDA)
        # get all source files in the current directory
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cu" )
        set(CUDA_SEPARABLE_COMPILATION OFF) # TODO required?
        add_fetched_tests_helper("${test_sources}" cuda GridToolsTestCUDA "${labels}")
    endif(GT_ENABLE_TARGET_CUDA)
endfunction(fetch_gpu_tests)

# This function can be used to add a custom x86 test
function(add_custom_x86_test)
    set(options )
    set(one_value_args TARGET)
    set(multi_value_args SOURCES COMPILE_DEFINITIONS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (___TARGET MATCHES "_x86$")
        message(WARNING "Test ${___TARGET} already has suffix _x86. Please remove suffix.")
    endif ()

    if (GT_ENABLE_TARGET_X86)
        set(name "${___TARGET}_x86")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_link_libraries(${name} gmock gtest_main GridToolsTestX86)
        target_compile_definitions(${name} PRIVATE ${___COMPILE_DEFINITIONS})
        add_test (NAME ${name} COMMAND ${exe} )
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
    endif (GT_ENABLE_TARGET_X86)
endfunction(add_custom_x86_test)

# This function can be used to add a custom mc test
function(add_custom_mc_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES COMPILE_DEFINITIONS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (___TARGET MATCHES "_mc$")
        message(WARNING "Test ${___TARGET} already has suffix _mc. Please remove suffix.")
    endif ()

    if (GT_ENABLE_TARGET_MC)
        set(name "${___TARGET}_mc")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_link_libraries(${name} gmock gtest_main GridToolsTestMC)
        target_compile_definitions(${name} PRIVATE ${___COMPILE_DEFINITIONS})
        add_test (NAME ${name} COMMAND ${exe} )
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
    endif (GT_ENABLE_TARGET_MC)
endfunction(add_custom_mc_test)

# This function can be used to add a custom gpu test
function(add_custom_gpu_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES COMPILE_DEFINITIONS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (___TARGET MATCHES "_cuda$")
        message(WARNING "Test ${___TARGET} already has suffix _cuda. Please remove suffix.")
    endif ()

    if (GT_ENABLE_TARGET_CUDA)
        set(name "${___TARGET}_cuda")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_link_libraries(${name} gmock gtest_main GridToolsTestCUDA)
        target_compile_definitions(${name} PRIVATE ${___COMPILE_DEFINITIONS})
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
        add_test (NAME ${name} COMMAND ${exe} )
    endif (GT_ENABLE_TARGET_CUDA)
endfunction(add_custom_gpu_test)


function(add_custom_mpi_x86_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES COMPILE_DEFINITIONS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (___TARGET MATCHES "_x86$")
        message(WARNING "Test ${___TARGET} already has suffix _x86. Please remove suffix.")
    endif ()

    if (GT_ENABLE_TARGET_X86)
        set(name "${___TARGET}_x86")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_link_libraries(${name} gmock mpi_gtest_main gcl GridToolsTestX86)
        target_compile_definitions(${name} PRIVATE ${___COMPILE_DEFINITIONS})
        gridtools_add_mpi_test(${name} ${exe})
    endif (GT_ENABLE_TARGET_X86)
endfunction(add_custom_mpi_x86_test)

function(add_custom_mpi_mc_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES COMPILE_DEFINITIONS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (___TARGET MATCHES "_mc$")
        message(WARNING "Test ${___TARGET} already has suffix _mc. Please remove suffix.")
    endif ()

    if (GT_ENABLE_TARGET_MC)
        set(name "${___TARGET}_mc")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_link_libraries(${name} gmock mpi_gtest_main gcl GridToolsTestMC)
        target_compile_definitions(${name} PRIVATE ${___COMPILE_DEFINITIONS})
        gridtools_add_mpi_test(${name} ${exe})
    endif (GT_ENABLE_TARGET_MC)
endfunction(add_custom_mpi_mc_test)

# This function can be used to add a custom gpu test
function(add_custom_mpi_gpu_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES COMPILE_DEFINITIONS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (___TARGET MATCHES "_cuda$")
        message(WARNING "Test ${___TARGET} already has suffix _cuda. Please remove suffix.")
    endif ()

    if (GT_ENABLE_TARGET_CUDA)
        set(name "${___TARGET}_cuda")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        #set(CUDA_SEPARABLE_COMPILATION OFF) TODO check this (remove or enable)
        add_executable (${name} ${___SOURCES} )
        target_link_libraries(${name} gmock mpi_gtest_main gcl GridToolsTestCUDA)
        target_compile_definitions(${name} PRIVATE ${___COMPILE_DEFINITIONS})
        gridtools_add_cuda_mpi_test(${name} ${exe})
    endif (GT_ENABLE_TARGET_CUDA)
endfunction(add_custom_mpi_gpu_test)
