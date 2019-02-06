// This file is generated!
#pragma once

#include <gridtools/c_bindings/array_descriptor.h>
#include <gridtools/c_bindings/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

gt_handle* make_copy_stencil(gt_handle*);
gt_handle* make_wrapper(int, int, int);
void run_stencil(gt_handle*, gt_handle*, gt_fortran_array_descriptor*, gt_fortran_array_descriptor*);

#ifdef __cplusplus
}
#endif
