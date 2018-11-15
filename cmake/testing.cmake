# This file will only be included when BUILD_TESTING = ON
# TODO move GridToolsTest target to this file
enable_testing()

####################################################################################
########################### GET GTEST LIBRARY ############################
####################################################################################

# include Threads manually before googletest such that we can properly apply the workaround
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package( Threads REQUIRED )
target_link_libraries( GridToolsTest INTERFACE Threads::Threads)
include(workaround_threads)
_fix_threads_flags()

add_subdirectory(./tools/googletest)

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

function (fetch_tests_helper target_arch filetype subfolder)
    set(options)
    set(one_value_args)
    set(multi_value_args LABELS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    string(TOLOWER ${target_arch} target_arch_l)
    string(TOUPPER ${target_arch} target_arch_u)

    set(labels ${___LABELS})
    list(APPEND labels target_${target_arch_l})

    if (GT_ENABLE_TARGET_${target_arch_u})
        # get all source files in the current directory
        file(GLOB test_sources "./${subfolder}/test_*.${filetype}" )
        foreach(test_source IN LISTS test_sources )
            # create a nice name for the test case
            get_filename_component (unit_test ${test_source} NAME_WE )
            set(unit_test "${unit_test}_${target_arch_l}")
            # create the test
            add_executable (${unit_test} ${test_source} )
            target_link_libraries(${unit_test} GridToolsTest${target_arch_u} c_bindings_generator c_bindings_handle gtest gmock_main)

            gridtools_add_test(
                NAME ${unit_test}
                SCRIPT ${TEST_SCRIPT}
                COMMAND $<TARGET_FILE:${unit_test}>
                LABELS ${labels})
        endforeach()
    endif()
endfunction()

# This function will fetch all x86 test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_x86_tests)
    fetch_tests_helper(x86 cpp ${ARGN})
endfunction(fetch_x86_tests)

# This function will fetch all mc test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_mc_tests)
    fetch_tests_helper(mc cpp ${ARGN})
endfunction(fetch_mc_tests)

# This function will fetch all gpu test cases in the given directory.
# Only used for nvcc compilations
function(fetch_gpu_tests)
    set(CUDA_SEPARABLE_COMPILATION OFF) # TODO required?
    fetch_tests_helper(cuda cu ${ARGN})
endfunction(fetch_gpu_tests)

function(add_custom_test_helper target_arch)
    set(options )
    set(one_value_args TARGET)
    set(multi_value_args SOURCES COMPILE_DEFINITIONS LABELS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    string(TOLOWER ${target_arch} target_arch_l)
    string(TOUPPER ${target_arch} target_arch_u)

    if (___TARGET MATCHES "_${target_arch_l}$")
        message(WARNING "Test ${___TARGET} already has suffix _${target_arch_l}. Please remove suffix.")
    endif ()

    set(labels ${___LABELS})
    list(APPEND labels target_${target_arch_l})

    if (GT_ENABLE_TARGET_${target_arch_u})
        set(unit_test "${___TARGET}_${target_arch_l}")
        # create the test
        add_executable (${unit_test} ${___SOURCES})
        target_link_libraries(${unit_test} gmock gtest_main GridToolsTest${target_arch_u})
        target_compile_definitions(${unit_test} PRIVATE ${___COMPILE_DEFINITIONS})
        gridtools_add_test(
            NAME ${unit_test}
            SCRIPT ${TEST_SCRIPT}
            COMMAND $<TARGET_FILE:${unit_test}>
            LABELS ${labels}
            )
    endif ()

endfunction()

# This function can be used to add a custom x86 test
function(add_custom_x86_test)
    add_custom_test_helper(x86 ${ARGN})
endfunction(add_custom_x86_test)

# This function can be used to add a custom mc test
function(add_custom_mc_test)
    add_custom_test_helper(mc ${ARGN})
endfunction(add_custom_mc_test)

# This function can be used to add a custom gpu test
function(add_custom_gpu_test)
    add_custom_test_helper(cuda ${ARGN})
endfunction(add_custom_gpu_test)

function(add_custom_test_helper target_arch)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES COMPILE_DEFINITIONS LABELS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    string(TOLOWER ${target_arch} target_arch_l)
    string(TOUPPER ${target_arch} target_arch_u)

    if (___TARGET MATCHES "_${target_arch_l}$")
        message(WARNING "Test ${___TARGET} already has suffix _${target_arch_l}. Please remove suffix.")
    endif ()

    set(labels ${___LABELS})
    list(APPEND labels target_${target_arch_l} )

    if (GT_ENABLE_TARGET_${target_arch_u})
        set(unit_test "${___TARGET}_${target_arch_l}")
        # create the test
        add_executable (${unit_test} ${___SOURCES})
        target_link_libraries(${unit_test} gmock mpi_gtest_main gcl GridToolsTest${target_arch_u})
        target_compile_definitions(${unit_test} PRIVATE ${___COMPILE_DEFINITIONS})
        gridtools_add_mpi_test(
            NAME ${unit_test}
            COMMAND $<TARGET_FILE:${unit_test}>
            LABELS ${labels}
            )
    endif ()
endfunction()

function(add_custom_mpi_x86_test)
    add_custom_test_helper(x86 ${ARGN})
endfunction(add_custom_mpi_x86_test)

function(add_custom_mpi_mc_test)
    add_custom_test_helper(mc ${ARGN})
endfunction(add_custom_mpi_mc_test)

# This function can be used to add a custom gpu test
function(add_custom_mpi_gpu_test)
    add_custom_test_helper(cuda ${ARGN})
endfunction(add_custom_mpi_gpu_test)
