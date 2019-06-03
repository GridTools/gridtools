# TODO rename this file

function(gt_add_bindings_library)
    message(DEPRECATION "gt_add_bindings_library() is deprecated: use cpp_bindgen_add_library().")
    cpp_bindgen_add_library(${ARGN})
endfunction()

function(gt_enable_bindings_library_fortran)
    message(DEPRECATION "gt_enable_bindings_library_fortran() is deprecated: use cpp_bindgen_enable_fortran_library().")
    cpp_bindgen_enable_fortran_library(${ARGN})
endfunction()
