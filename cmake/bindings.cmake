function(generate_gt_bindings)
    set(BINDINGS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
    set(BINDINGS_GENERATE_PATH ${CMAKE_SOURCE_DIR}/cmake/)
    set(BINDINGS_LIBRARIES GridTools)
    configure_file(cmake/gt_bindings.cmake.in
        ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/bindings_for_build/gt_bindings.cmake
        @ONLY)
endfunction()

generate_gt_bindings()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/bindings_for_build")
include(gt_bindings)
