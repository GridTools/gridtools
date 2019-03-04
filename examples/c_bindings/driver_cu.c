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

#include "copy_stencil_lib_cu.h"
#include "cuda_runtime.h"

int main() {
    gt_handle *wrapper_handle = make_wrapper(9, 10, 11);
    gt_handle *computation_handle = make_copy_stencil(wrapper_handle);

    // Note that the order of the indices array here is k, j, i (which is the internal
    // Fortran layout). This is the layout that is expected by the bindings we have
    // written.
    float *in_array, *out_array;
    cudaMallocManaged((void **)&in_array, 9 * 10 * 11 * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&out_array, 9 * 10 * 11 * sizeof(float), cudaMemAttachGlobal);

    // fill some inputs
    float n1 = 0;
    float n2 = 11 * 10 * 9;
    for (int k = 0; k < 11; ++k)
        for (int j = 0; j < 10; ++j)
            for (int i = 0; i < 9; ++i, ++n1, --n2) {
                in_array[k * 10 * 9 + j * 9 + i] = n1;
                out_array[k * 10 * 9 + j * 9 + i] = n2;
            }

    // in the C bindings, the fortran array descriptors need to be filled explicitly
    gt_fortran_array_descriptor in_descriptor = {
        .rank = 3, .type = gt_fk_Float, .dims = {9, 10, 11}, .data = (void *)in_array};
    gt_fortran_array_descriptor out_descriptor = {
        .rank = 3, .type = gt_fk_Float, .dims = {9, 10, 11}, .data = (void *)out_array};

    run_stencil(wrapper_handle, computation_handle, &in_descriptor, &out_descriptor);

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
    return 0;
}
