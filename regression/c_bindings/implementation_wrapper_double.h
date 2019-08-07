// This file is generated!
#pragma once

#include <cpp_bindgen/array_descriptor.h>
#include <cpp_bindgen/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

bindgen_handle *create_copy_stencil(bindgen_fortran_array_descriptor *, bindgen_fortran_array_descriptor *);
void run_stencil(bindgen_handle *);
void sync_data_store(bindgen_fortran_array_descriptor *);

#ifdef __cplusplus
}
#endif
