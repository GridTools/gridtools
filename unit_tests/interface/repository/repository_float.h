// This file is generated!
#pragma once

#include <cpp_bindgen/array_descriptor.h>
#include <cpp_bindgen/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

gen_handle* make_exported_repository(int, int, int);
void prefix_set_exported_ijfield(gen_handle*, gen_fortran_array_descriptor*);
void prefix_set_exported_ijkfield(gen_handle*, gen_fortran_array_descriptor*);
void prefix_set_exported_jkfield(gen_handle*, gen_fortran_array_descriptor*);
void verify_exported_repository(gen_handle*);

#ifdef __cplusplus
}
#endif
