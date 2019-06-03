// This file is generated!
#pragma once

#include <cpp_bindgen/array_descriptor.h>
#include <cpp_bindgen/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

gen_handle* create_copy_stencil(gen_fortran_array_descriptor*, gen_fortran_array_descriptor*);
void run_stencil(gen_handle*);
void sync_data_store(gen_fortran_array_descriptor*);

#ifdef __cplusplus
}
#endif
