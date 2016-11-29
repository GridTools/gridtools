project(Storage)
cmake_minimum_required(VERSION 2.8)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# load some cmake includes
include (flags)
include (definitions)

# look for source files:
# ======================
#    NOTE: this thing has only one main function in the main.cpp file
#           in the case that you build another program, you migh need to 
#           pass that file to the add_executable command, as another parameter
#           I know 2 solutions: 
#               - change name of main files to extension cxx and include them manually in add_executable
#               - remove main files form the sources variable and add them manually in add_executable
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" )



#   Configure build 
#   ===============
#   include path, compiler flags
include_directories ( "include/" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread")
get_filename_component(FN ${sources} NAME_WE)
add_executable( ${FN} ${sources} )


# tests:
# ======

include (testing)

## gcc tests:
file(GLOB_RECURSE test_sources "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
add_definitions(-DGTEST_COLOR ) 
include_directories( ${GTEST_INCLUDE_DIR} )
include_directories( ${GMOCK_INCLUDE_DIR} )
foreach( test_source ${test_sources} )
    
    get_filename_component (unit_test ${test_source} NAME_WE )
    get_filename_component(P_DIR ${test_source} DIRECTORY)
    get_filename_component(PP_DIR ${P_DIR} DIRECTORY)

    string(REPLACE ${PP_DIR} ${CMAKE_BINARY_DIR} DIR ${P_DIR})
    string(REPLACE ${PP_DIR}/ "" FOLDER ${P_DIR})

    if (${PP_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
        if (${test_source} MATCHES ".*.cu" AND ${USE_GPU})
            set (unit_test_trg ${unit_test}_cuda)
            set (unit_test ${unit_test}_cuda)            
        elseif(${test_source} MATCHES ".*.cpp")
            set (unit_test_trg ${unit_test}_host)
            set (unit_test ${unit_test}_host)
        endif()
        set (FOLDER ".")
    else()
        set (unit_test_trg ${FOLDER}_${unit_test})
    endif()

    if (${test_source} MATCHES ".*.cu" AND ${USE_GPU})
        cuda_add_executable (${unit_test_trg} ${test_source})
        target_link_libraries(${unit_test_trg} gtest gtest_main cudart pthread)
        set_target_properties(${unit_test_trg} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${DIR} OUTPUT_NAME ${unit_test})
        add_test (${FOLDER}/${unit_test} ${DIR}/${unit_test} )
    elseif(${test_source} MATCHES ".*.cpp")
        add_executable (${unit_test_trg} ${test_source})
        target_link_libraries(${unit_test_trg} gtest gtest_main pthread)
        set_target_properties(${unit_test_trg} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${DIR} OUTPUT_NAME ${unit_test})
        add_test (${FOLDER}/${unit_test} ${DIR}/${unit_test} )
    endif()

endforeach(test_source)

enable_testing()
