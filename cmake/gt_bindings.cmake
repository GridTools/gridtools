message(DEPRECATION "Including gt_bindings.cmake is deprecated use find_package(cpp_bindgen)")
include(${cpp_bindgen_CMAKE_DIR}/cpp_bindgen.cmake)

add_library(c_bindings_generator ALIAS cpp_bindgen_generator)

function(gt_add_bindings_library)
    message(DEPRECATION "gt_add_bindings_library() is deprecated: use bindgen_add_library().")
    bindgen_add_library(${ARGN})
endfunction()

function(gt_enable_bindings_library_fortran)
    message(DEPRECATION "gt_enable_bindings_library_fortran() is deprecated: use bindgen_enable_fortran_library().")
    bindgen_enable_fortran_library(${ARGN})
endfunction()
