cmake_minimum_required(VERSION 2.8.8)
enable_testing()

####################################################################################
########################### GET GTEST AND GMOCK LIBRARY ############################ 
####################################################################################

# We need thread support
find_package(Threads REQUIRED)

# Enable ExternalProject CMake module
include(ExternalProject)

# Download and install GoogleTest
ExternalProject_Add(
    gtest
    URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # forward toolchain
    CMAKE_ARGS 
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_C_COMPILER_ARG1=${CMAKE_C_COMPILER_ARG1}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_COMPILER_ARG1=${CMAKE_CXX_COMPILER_ARG1}
    # Disable install step
    INSTALL_COMMAND ""
)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_library(libgtest_main IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)
add_dependencies(libgtest_main gtest)

# Set gtest properties
ExternalProject_Get_Property(gtest source_dir binary_dir)
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)
set_target_properties(libgtest_main PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/libgtest_main.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

include_directories("${source_dir}/include")

# Download and install GoogleMock
ExternalProject_Add(
    gmock
    URL https://googlemock.googlecode.com/files/gmock-1.7.0.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gmock
    # forward toolchain
    CMAKE_ARGS 
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_C_COMPILER_ARG1=${CMAKE_C_COMPILER_ARG1}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_COMPILER_ARG1=${CMAKE_CXX_COMPILER_ARG1}
    # Disable install step
    INSTALL_COMMAND ""
)

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gmock)

# Set gmock properties
ExternalProject_Get_Property(gmock source_dir binary_dir)
set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/libgmock.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)
include_directories("${source_dir}/include")

####################################################################################
######################### ADDITIONAL TEST MODULE FUNCTIONS ######################### 
####################################################################################

# This function will fetch all test cases in the given directory.
# Only used for gcc or clang compilations
function(fetch_host_tests subfolder)
    # get all source files in the current directory
    file(GLOB test_sources_cxx03 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx03_*.cpp" )
    file(GLOB test_sources_cxx11 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx11_*.cpp" )
    file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cpp" )

    # remove files that should not be there
    if(ENABLE_CXX11)
        foreach( cxx03 ${test_sources_cxx03} )
            list(REMOVE_ITEM test_sources ${cxx03})
        endforeach()
    else()
        foreach( cxx11 ${test_sources_cxx11} )
            list(REMOVE_ITEM test_sources ${cxx11})
        endforeach()
    endif()

    # set include dirs
    include_directories( ${GTEST_INCLUDE_DIR} )
	include_directories( ${GMOCK_INCLUDE_DIR} )
	# add definitions
	add_definitions(-DGTEST_COLOR )

    # create all targets
    foreach( test_source ${test_sources} )
        # create a nice name for the test case
        get_filename_component (unit_test ${test_source} NAME_WE )
        set(unit_test ${unit_test})
        # set binary output name and dir
        set(exe ${CMAKE_CURRENT_BINARY_DIR}/${unit_test})
        # create the test
        add_executable (${unit_test} ${test_source})
        target_link_libraries(${unit_test} ${exe_LIBS} libgtest libgtest_main libgmock)
        add_test (NAME ${unit_test} COMMAND ${exe} )
        gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})        
        message( "added test " ${unit_test} )         
    endforeach(test_source)
endfunction(fetch_host_tests)  

# This function will fetch all gpu test cases in the given directory.
# Only used for nvcc compilations
function(fetch_gpu_tests subfolder)
    # don't care if USE_GPU is not set
    if(USE_GPU)
        # get all source files in the current directory
        file(GLOB test_sources_cxx03 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx03_*.cu" )
        file(GLOB test_sources_cxx11 "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_cxx11_*.cu" )
        file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${subfolder}/test_*.cu" )

        # remove files that should not be there
        if(ENABLE_CXX11)
            foreach( cxx03 ${test_sources_cxx03} )
                list(REMOVE_ITEM test_sources ${cxx03})
            endforeach()
        else()
            foreach( cxx11 ${test_sources_cxx11} )
                list(REMOVE_ITEM test_sources ${cxx11})
            endforeach()
        endif()

        # set include dirs
        include_directories( ${GTEST_INCLUDE_DIR} )
        include_directories( ${GMOCK_INCLUDE_DIR} )

        # add definitions
        add_definitions(-DGTEST_COLOR )

        # create all targets
        foreach( test_source ${test_sources} )
            # create a nice name for the test case
            get_filename_component (filename ${test_source} NAME_WE )
            set(postfix "_cuda")
            set(unit_test ${filename}${postfix})
            # set binary output name and dir
            set(exe ${CMAKE_CURRENT_BINARY_DIR}/${unit_test})
            # create the gpu test
            cuda_add_executable (${unit_test} ${test_source})
            set_target_properties(${unit_test} PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS} LINKER_LANGUAGE CXX )            
            target_link_libraries(${unit_test} ${exe_LIBS} libgtest libgtest_main libgmock)
            gridtools_add_test(${unit_test} ${TEST_SCRIPT} ${exe})
            message( "added gpu test " ${unit_test} )         
        endforeach(test_source)
    endif(USE_GPU)
endfunction(fetch_gpu_tests)  

# This function can be used to add a custom host test
function(add_custom_host_test name sources cc_flags ld_flags)
    # set include dirs
    include_directories( ${GTEST_INCLUDE_DIR} )
	include_directories( ${GMOCK_INCLUDE_DIR} )
	# add definitions
	add_definitions(-DGTEST_COLOR )

    # set binary output name and dir
    set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
    # create the test
    add_executable (${name} ${sources})
    set(cflags "${cc_flags} ${CMAKE_CXX_FLAGS}" )
    set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cflags}" LINK_FLAGS ${ld_flags} LINKER_LANGUAGE CXX )            
    target_link_libraries(${name} ${exe_LIBS} libgtest libgtest_main libgmock)
    add_test (NAME ${name} COMMAND ${exe} )
    gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
endfunction(add_custom_host_test) 

# This function can be used to add a custom gpu test
function(add_custom_gpu_test name sources cc_flags ld_flags)
    # set include dirs
    include_directories( ${GTEST_INCLUDE_DIR} )
	include_directories( ${GMOCK_INCLUDE_DIR} )
	# add definitions
	add_definitions(-DGTEST_COLOR )

    # set binary output name and dir
    set(exe ${CMAKE_CURRENT_BINARY_DIR}/${name})
    # create the test
    cuda_add_executable (${name} ${test_source})
    set(cflags ${CMAKE_CXX_FLAGS} ${cc_flags})
    set_target_properties(${name} PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS} "${cflags}" LINK_FLAGS "${ld_flags}" LINKER_LANGUAGE CXX )            
    target_link_libraries(${name} ${exe_LIBS} libgtest libgtest_main libgmock)
    gridtools_add_test(${name} ${TEST_SCRIPT} ${exe})
endfunction(add_custom_gpu_test) 