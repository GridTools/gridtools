// This file is generated!
#pragma once

#include <gridtools/c_bindings/array_descriptor.h>
#include <gridtools/c_bindings/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

gt_handle* create_copy_stencil(gt_handle*, gt_handle*);
gt_handle* create_data_store(unsigned int, unsigned int, unsigned int, double*);
gt_handle* generic_create_data_store0(unsigned int, unsigned int, unsigned int, double*);
gt_handle* generic_create_data_store1(unsigned int, unsigned int, unsigned int, float*);
void run_stencil(gt_handle*);
void sync_data_store(gt_handle*);

#ifdef __cplusplus
}
#endif
