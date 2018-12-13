// This file is generated!
#pragma once

#include <gridtools/c_bindings/array_descriptor.h>
#include <gridtools/c_bindings/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

gt_handle* make_exported_repository(int, int, int);
void set_exported_ijfield(gt_handle*, gt_fortran_array_descriptor*);
void set_exported_ijkfield(gt_handle*, gt_fortran_array_descriptor*);
void set_exported_jkfield(gt_handle*, gt_fortran_array_descriptor*);
void verify_exported_repository(gt_handle*);

#ifdef __cplusplus
}
#endif
