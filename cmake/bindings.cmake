function(generate_gt_bindings)
    set(BINDINGS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
    set(BINDINGS_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake/)
    configure_file(cmake/gt_bindings.cmake.in
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/bindings_for_build/gt_bindings.cmake
        @ONLY)
endfunction()

generate_gt_bindings()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/bindings_for_build")
include(gt_bindings)
