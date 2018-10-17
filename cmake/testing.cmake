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
add_library(gtest_main ${CMAKE_CURRENT_SOURCE_DIR}/tools/googletest/googletest/src/gtest_main.cc)
if( NOT GCL_ONLY )
    if( USE_MPI )
        if ( ENABLE_CUDA )
            include_directories ( "${CUDA_INCLUDE_DIRS}" )
        endif()
        add_library( mpi_gtest_main include/gridtools/tools/mpi_unit_test_driver/mpi_test_driver.cpp )
        set_target_properties(mpi_gtest_main PROPERTIES COMPILE_FLAGS "${GPU_SPECIFIC_FLAGS}" )
        #target_link_libraries(mpi_gtest_main ${exe_LIBS} )
    endif()
endif()
set( exe_LIBS ${exe_LIBS} gtest)

####################################################################################
######################### ADDITIONAL TEST MODULE FUNCTIONS #########################
####################################################################################

# This function will fetch all host test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_host_tests subfolder)
    if (ENABLE_HOST)
        # get all source files in the current directory
        file(GLOB test_sources_cxx11 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx11_*.cpp" )
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cpp" )
        file(GLOB test_headers "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.hpp" )

        # create all targets
        foreach( test_source ${test_sources} )
            # create a nice name for the test case
            get_filename_component (unit_test ${test_source} NAME_WE )
            set(unit_test "${unit_test}_host")
            # set binary output name and dir
            set(exe ${CMAKE_CURRENT_BINARY_DIR}/${unit_test})
            # create the test
            add_executable (${unit_test} ${test_source} ${test_headers})
            target_link_libraries(${unit_test} ${exe_LIBS} gtest_main )
            target_compile_options(${unit_test} PUBLIC ${GT_CXX_FLAGS} -D${HOST_BACKEND_DEFINE})
            target_include_directories(${unit_test}
                 PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
            )
            add_test (NAME ${unit_test} COMMAND ${exe} )
            gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})
            # message( "added test " ${unit_test} )
        endforeach(test_source)
    endif(ENABLE_HOST)
endfunction(fetch_host_tests)

# This function will fetch all mic test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_mic_tests subfolder)
    if (ENABLE_MIC)
        # get all source files in the current directory
        file(GLOB test_sources_cxx11 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx11_*.cpp" )
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cpp" )
        file(GLOB test_headers "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/*.hpp" )

        # create all targets
        foreach( test_source ${test_sources} )
            # create a nice name for the test case
            get_filename_component (unit_test ${test_source} NAME_WE )
            set(unit_test "${unit_test}_mic")
            # set binary output name and dir
            set(exe ${CMAKE_CURRENT_BINARY_DIR}/${unit_test})
            # create the test
            add_executable (${unit_test} ${test_source} ${test_headers})
            target_link_libraries(${unit_test} ${exe_LIBS} gtest_main )
            target_compile_options(${unit_test} PUBLIC ${GT_CXX_FLAGS} -D${MIC_BACKEND_DEFINE})
            target_include_directories(${unit_test}
                 PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
            )
            add_test (NAME ${unit_test} COMMAND ${exe} )
            gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})
            # message( "added test " ${unit_test} )
        endforeach(test_source)
    endif(ENABLE_MIC)
endfunction(fetch_mic_tests)

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
            cuda_add_executable (${unit_test} ${test_source} ${test_headers} OPTIONS "${GT_CXX_FLAGS} ${GPU_SPECIFIC_FLAGS} -D${CUDA_BACKEND_DEFINE}")
            target_link_libraries(${unit_test}  gtest_main ${exe_LIBS} )
            target_include_directories(${unit_test}
                 PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
            )
            gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})
            # message( "added gpu test " ${unit_test} )
        endforeach(test_source)
    endif(ENABLE_CUDA)
endfunction(fetch_gpu_tests)

# This function can be used to add a custom host test
function(add_custom_host_test)
    set(options )
    set(one_value_args TARGET)
    set(multi_value_args SOURCES ADDITIONAL_FLAGS)
    cmake_parse_arguments(HT "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ENABLE_HOST)
        set(name "${HT_TARGET}_host")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${HT_SOURCES})
        target_compile_options(${name} PUBLIC ${GT_CXX_FLAGS} -D${HOST_BACKEND_DEFINE} ${HT_ADDITIONAL_FLAGS})
        target_link_libraries(${name} ${exe_LIBS} gtest_main)
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        add_test (NAME ${name} COMMAND ${exe} )
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
    endif (ENABLE_HOST)
endfunction(add_custom_host_test)

# This function can be used to add a custom mic test
function(add_custom_mic_test)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES ADDITIONAL_FLAGS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ENABLE_MIC)
        set(name "${___TARGET}_mic")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${___SOURCES})
        target_compile_options(${name} PUBLIC ${GT_CXX_FLAGS} -D${MIC_BACKEND_DEFINE} ${___ADDITIONAL_FLAGS})
        target_link_libraries(${name} ${exe_LIBS} gtest_main)
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        add_test (NAME ${name} COMMAND ${exe} )
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
    endif (ENABLE_MIC)
endfunction(add_custom_mic_test)

# This function can be used to add a custom gpu test
function(add_custom_gpu_test name sources cc_flags ld_flags)
    if (ENABLE_CUDA)
        set(name "${name}_cuda")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        set(CUDA_SEPARABLE_COMPILATION OFF)
        cuda_add_executable (${name} ${test_source} OPTIONS "${GT_CXX_FLAGS} -D${CUDA_BACKEND_DEFINE}")
        set(cflags ${CMAKE_CXX_FLAGS} ${cc_flags} COMPILE_FLAGS ${GPU_SPECIFIC_FLAGS} "${cflags}" LINK_FLAGS "${ld_flags}" LINKER_LANGUAGE CXX)
        target_link_libraries(${name} ${exe_LIBS} gtest_main)
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
    endif (ENABLE_CUDA)
endfunction(add_custom_gpu_test)


function(add_custom_mpi_host_test name sources cc_flags ld_flags)
    if (ENABLE_HOST)
        set(name "${name}_host")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${sources})
        target_compile_options(${name} PUBLIC ${GT_CXX_FLAGS} ${cc_flags} -D${HOST_BACKEND_DEFINE})
        target_link_libraries(${name} mpi_gtest_main ${exe_LIBS})
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        gridtools_add_mpi_test(${name} ${exe})
    endif (ENABLE_HOST)
endfunction(add_custom_mpi_host_test)

function(add_custom_mpi_mic_test name sources cc_flags ld_flags)
    if (ENABLE_MIC)
        set(name "${name}_mic")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        add_executable (${name} ${sources})
        target_link_libraries(${name} mpi_gtest_main ${exe_LIBS})
        target_compile_options(${name} PUBLIC ${GT_CXX_FLAGS} ${cc_flags} -D${MIC_BACKEND_DEFINE})
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        gridtools_add_mpi_test(${name} ${exe})
    endif (ENABLE_MIC)
endfunction(add_custom_mpi_mic_test)

# This function can be used to add a custom gpu test
function(add_custom_mpi_gpu_test name sources cc_flags ld_flags)
    if (ENABLE_CUDA)
        set(name "${name}_cuda")
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
        # create the test
        #set(CUDA_SEPARABLE_COMPILATION OFF)
        cuda_add_executable (${name} ${sources} OPTIONS ${GPU_SPECIFIC_FLAGS} ${cc_flags} "-D${CUDA_BACKEND_DEFINE}")
        set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cflags} ${GPU_SPECIFIC_FLAGS}" )
        target_link_libraries(${name} ${exe_LIBS} mpi_gtest_main)
        target_include_directories(${name}
             PRIVATE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        )
        gridtools_add_cuda_mpi_test(${name} ${exe})
    endif (ENABLE_CUDA)
endfunction(add_custom_mpi_gpu_test)
