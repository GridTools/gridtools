// This file is generated!
#pragma once

#include <gridtools/c_bindings/array_descriptor.h>
#include <gridtools/c_bindings/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

gt_handle* create_copy_stencil(gt_fortran_array_descriptor*, gt_fortran_array_descriptor*);
void run_stencil(gt_handle*);
void sync_data_store(gt_fortran_array_descriptor*);

#ifdef __cplusplus
}
#endif
