/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>

#include "copy_stencil_lib_mc.h"

int main() {
    gt_handle *grid_handle = make_grid(9, 10, 11);
    gt_handle *storage_info_handle = make_storage_info(9, 10, 11);
    gt_handle *in_handle = make_data_store(storage_info_handle);
    gt_handle *out_handle = make_data_store(storage_info_handle);
    gt_handle *computation_handle = make_copy_stencil(grid_handle);
    free(storage_info_handle);
    free(grid_handle);

    // Note that the order of the indices array here is k, j, i (which is the internal
    // Fortran layout). This is the layout that is expected by the bindings we have
    // written.
    float *in_array = (float *)(malloc(11 * 10 * 9 * sizeof(float)));
    float *out_array = (float *)(malloc(11 * 10 * 9 * sizeof(float)));

    // fill some inputs
    float n1 = 0;
    float n2 = 11 * 10 * 9;
    for (int k = 0; k < 11; ++k)
        for (int j = 0; j < 10; ++j)
            for (int i = 0; i < 9; ++i, ++n1, --n2) {
                int index = k * 10 * 9 + j * 9 + i;
                in_array[index] = n1;
                out_array[index] = n2;
            }

    // in the C bindings, the fortran array descriptors need to be filled explicitly
    gt_fortran_array_descriptor in_descriptor = {
        .rank = 3, .type = gt_fk_Float, .dims = {9, 10, 11}, .data = (void *)in_array};
    gt_fortran_array_descriptor out_descriptor = {
        .rank = 3, .type = gt_fk_Float, .dims = {9, 10, 11}, .data = (void *)out_array};

    transform_f_to_c(in_handle, &in_descriptor);
    run_stencil(computation_handle, in_handle, out_handle);
    transform_c_to_f(&out_descriptor, out_handle);

    // now, the output can be verified
    for (int k = 0; k < 11; ++k)
        for (int j = 0; j < 10; ++j)
            for (int i = 0; i < 9; ++i) {
                int idx = k * 10 * 9 + j * 9 + i;
                if (in_array[idx] != out_array[idx]) {
                    printf("Error at position (k=%i, j=%i, i=%i)\n", k, j, i);
                    return 1;
                }
            }

    printf("It works!\n");

    free(in_handle);
    free(out_handle);
    free(computation_handle);
    free(in_array);
    free(out_array);

    return 0;
}
