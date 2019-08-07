// This file is generated!
#pragma once

#include <cpp_bindgen/array_descriptor.h>
#include <cpp_bindgen/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

bindgen_handle *make_exported_repository(int, int, int);
void prefix_set_exported_ijfield(bindgen_handle *, bindgen_fortran_array_descriptor *);
void prefix_set_exported_ijkfield(bindgen_handle *, bindgen_fortran_array_descriptor *);
void prefix_set_exported_jkfield(bindgen_handle *, bindgen_fortran_array_descriptor *);
void verify_exported_repository(bindgen_handle *);

#ifdef __cplusplus
}
#endif
