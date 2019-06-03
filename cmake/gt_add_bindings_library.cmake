function(gt_add_bindings_library)
    message(DEPRECATION "gt_add_bindings_library() is deprecated: use cpp_bindgen_add_library().")
    cpp_bindgen_add_library(${ARGN})
endfunction()
