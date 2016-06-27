project(GridTools)
cmake_minimum_required(VERSION 2.8)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ===============
# Load some cmake modules.
# 1) define what flags should be available, 
# 2) package by name finder
# 3) testing methods
# 4) build configurator (include path, compiler flags, etc.
# ===============
include (flags)
include (testing)
include (definitions)

# ===============
# library header files
# ===============
include_directories ( "${CMAKE_CURRENT_SOURCE_DIR}/include" )
include_directories ( "${CMAKE_CURRENT_SOURCE_DIR}/include/communication" )
include_directories ( "${CMAKE_CURRENT_SOURCE_DIR}/include/communication/high-level" )

# ===============
# compile gcl:
# ===============
add_library( gcl src/GCL.cpp include/communication/GCL.hpp include/communication/high-level/stats_collector.hpp )
set_target_properties(gcl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS}" )
set(exe_LIBS ${exe_LIBS} gcl)

# ===============
# add all cpp files in src:
# ===============
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" )
list(REMOVE_ITEM sources "${CMAKE_CURRENT_SOURCE_DIR}/src/GCL.cpp")
foreach( src_file ${sources} )
    get_filename_component (exec ${src_file} NAME_WE )
    add_executable (${exec} ${src_file})
    target_link_libraries(${exec} ${exe_LIBS})    
endforeach(src_file)

# ===============
# add all cpp files in examples:
# ===============
add_subdirectory(examples)

# ===============
# tests
# ===============
include (fix_test_case_name)
add_subdirectory(unit_tests)
