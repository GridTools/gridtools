cmake_minimum_required(VERSION 2.8.8)
enable_testing()

####################################################################################
########################### GET GTEST LIBRARY ############################
####################################################################################
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/tools/googletest/googletest/)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/tools/googletest/googletest/include)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/tools/googletest/googlemock/include)

# ===============
add_library(gtest ${CMAKE_CURRENT_SOURCE_DIR}/tools/googletest/googletest/src/gtest-all.cc)
target_compile_options( gtest PRIVATE ${GT_CXX_FLAGS} -std=c++11)
add_library(gtest_main ${CMAKE_CURRENT_SOURCE_DIR}/tools/googletest/googletest/src/gtest_main.cc)
target_compile_options( gtest_main PRIVATE ${GT_CXX_FLAGS} -std=c++11 )
if( NOT GCL_ONLY )
    if( USE_MPI )
        if ( ENABLE_CUDA )
            include_directories ( "${CUDA_INCLUDE_DIRS}" )
        endif()
        add_library( mpi_gtest_main include/gridtools/tools/mpi_unit_test_driver/mpi_test_driver.cpp )
        target_compile_options( mpi_gtest_main PRIVATE ${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS} -std=c++11)
    endif()
endif()
set( exe_LIBS ${exe_LIBS} gtest)

####################################################################################
######################### ADDITIONAL TEST MODULE FUNCTIONS #########################
####################################################################################

# This function will fetch all x86 test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_x86_tests subfolder)
    if (ENABLE_X86)
        # get all source files in the current directory
        file(GLOB test_sources_cxx11 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx11_*.cpp" )
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cpp" )
        file(GLOB test_headers "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.hpp" )

        # create all targets
        foreach( test_source ${test_sources} )
            # create a nice name for the test case
            get_filename_component (unit_test ${test_source} NAME_WE )
            set(unit_test "${unit_test}_x86")
            # set binary output name and dir
            set(exe ${CMAKE_CURRENT_BINARY_DIR}/${unit_test})
            # create the test
            add_executable (${unit_test} ${test_source} ${test_headers})
            target_link_libraries(${unit_test} ${exe_LIBS} ${OMP_LIBS} gtest_main )
            target_compile_options(${unit_test} PUBLIC ${GT_CXX_FLAGS} -D${X86_BACKEND_DEFINE})
            target_include_directories(${unit_test}
                 PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
            )
            add_test (NAME ${unit_test} COMMAND ${exe} )
            gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})
            # message( "added test " ${unit_test} )
        endforeach(test_source)
    endif(ENABLE_X86)
endfunction(fetch_x86_tests)

# This function will fetch all mc test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_mc_tests subfolder)
    if (ENABLE_MC)
        # get all source files in the current directory
        file(GLOB test_sources_cxx11 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx11_*.cpp" )
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cpp" )
        file(GLOB test_headers "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.hpp" )

        # create all targets
        foreach( test_source ${test_sources} )
            # create a nice name for the test case
            get_filename_component (unit_test ${test_source} NAME_WE )
            set(unit_test "${unit_test}_mc")
            # set binary output name and dir
            set(exe ${CMAKE_CURRENT_BINARY_DIR}/${unit_test})
            # create the test
            add_executable (${unit_test} ${test_source} ${test_headers})
            target_link_libraries(${unit_test} ${exe_LIBS} ${OMP_LIBS} gtest_main )
            target_compile_options(${unit_test} PUBLIC ${GT_CXX_FLAGS} -D${MC_BACKEND_DEFINE})
            target_include_directories(${unit_test}
                 PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
            )
            add_test (NAME ${unit_test} COMMAND ${exe} )
            gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})
            # message( "added test " ${unit_test} )
        endforeach(test_source)
    endif(ENABLE_MC)
endfunction(fetch_mc_tests)

# This function will fetch all gpu test cases in the given directory.
# Only used for nvcc compilations
function(fetch_gpu_tests subfolder)
    if(ENABLE_CUDA)
        # get all source files in the current directory
        file(GLOB test_sources_cxx11 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx11_*.cu" )
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cu" )
        file(GLOB test_headers "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.hpp" )

        # create all targets
        foreach( test_source ${test_sources} )
            # create a nice name for the test case
            get_filename_component (filename ${test_source} NAME_WE )
            set(unit_test "${filename}_cuda")
            # set binary output name and dir
            set(exe ${CMAKE_CURRENT_BINARY_DIR}/${unit_test})
            # create the gpu test
            set(CUDA_SEPARABLE_COMPILATION OFF)
            add_executable( ${unit_test} ${test_source} ${test_headers} )
            target_compile_options (${unit_test} PUBLIC ${GT_CXX_FLAGS} ${GT_CUDA_FLAGS} ${GPU_SPECIFIC_FLAGS} -D${CUDA_BACKEND_DEFINE})
            target_link_libraries(${unit_test}  ${exe_LIBS} ${OMP_LIBS} gtest_main )
            target_include_directories(${unit_test}
                 PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
            )
            gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})
            # message( "added gpu test " ${unit_test} )
        endforeach(test_source)
    endif(ENABLE_CUDA)
endfunction(fetch_gpu_tests)

# This function can be used to add a custom x86 test
function(add_custom_x86_test)
    set(options )
    set(one_value_args TARGET)
    set(multi_value_args SOURCES ADDITIONAL_FLAGS)
    cmake_parse_arguments(HT "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ENABLE_X86)
        set(name "${HT_TARGET}_x86")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${HT_SOURCES})
        target_compile_options(${name} PUBLIC ${GT_CXX_FLAGS} -D${X86_BACKEND_DEFINE} ${HT_ADDITIONAL_FLAGS})
        target_link_libraries(${name} ${exe_LIBS} ${OMP_LIBS} gtest_main)
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        add_test (NAME ${name} COMMAND ${exe} )
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
    endif (ENABLE_X86)
endfunction(add_custom_x86_test)

# This function can be used to add a custom mc test
function(add_custom_mc_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES ADDITIONAL_FLAGS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ENABLE_MC)
        set(name "${___TARGET}_mc")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_compile_options(${name} PUBLIC ${GT_CXX_FLAGS} -D${MC_BACKEND_DEFINE} ${___ADDITIONAL_FLAGS})
        target_link_libraries(${name} ${exe_LIBS} ${OMP_LIBS} gtest_main)
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        add_test (NAME ${name} COMMAND ${exe} )
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
    endif (ENABLE_MC)
endfunction(add_custom_mc_test)

# This function can be used to add a custom gpu test
function(add_custom_gpu_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES ADDITIONAL_FLAGS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ENABLE_CUDA)
        set(name "${___TARGET}_cuda")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        set(CUDA_SEPARABLE_COMPILATION OFF)
        add_executable (${name} ${___SOURCES})
        target_compile_options (${name} PUBLIC ${GT_CUDA_FLAGS} ${GT_CXX_FLAGS} -D${CUDA_BACKEND_DEFINE} ${___ADDITIONAL_FLAGS})

        target_link_libraries(${name} ${exe_LIBS} ${OMP_LIBS} gtest_main)
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
    endif (ENABLE_CUDA)
endfunction(add_custom_gpu_test)


function(add_custom_mpi_x86_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES ADDITIONAL_FLAGS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ENABLE_X86)
        set(name "${___TARGET}_x86")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_compile_options(${name} PUBLIC ${GT_CXX_FLAGS} ${___ADDITIONAL_FLAGS} -D${X86_BACKEND_DEFINE} )
        target_link_libraries(${name} mpi_gtest_main ${exe_LIBS} ${OMP_LIBS} )
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        gridtools_add_mpi_test(${name} ${exe})
    endif (ENABLE_X86)
endfunction(add_custom_mpi_x86_test)

function(add_custom_mpi_mc_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES ADDITIONAL_FLAGS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ENABLE_MC)
        set(name "${___TARGET}_mc")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_link_libraries(${name} mpi_gtest_main ${exe_LIBS} ${OMP_LIBS} )
        target_compile_options(${name} PUBLIC ${GT_CXX_FLAGS} ${___ADDITIONAL_FLAGS} -D${MC_BACKEND_DEFINE})
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        gridtools_add_mpi_test(${name} ${exe})
    endif (ENABLE_MC)
endfunction(add_custom_mpi_mc_test)

# This function can be used to add a custom gpu test
function(add_custom_mpi_gpu_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES ADDITIONAL_FLAGS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ENABLE_CUDA)
        set(name "${___TARGET}_cuda")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        #set(CUDA_SEPARABLE_COMPILATION OFF)
        add_executable (${name} ${___SOURCES} )
        target_compile_options (${name} PUBLIC ${GPU_SPECIFIC_FLAGS} ${GT_CXX_FLAGS} ${GT_CUDA_FLAGS} ${___ADDITIONAL_FLAGS} -D${CUDA_BACKEND_DEFINE})

        target_link_libraries(${name} ${exe_LIBS} ${OMP_LIBS} mpi_gtest_main)
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        gridtools_add_cuda_mpi_test(${name} ${exe})
    endif (ENABLE_CUDA)
endfunction(add_custom_mpi_gpu_test)
