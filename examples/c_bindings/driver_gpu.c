/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>

#include "copy_stencil_lib_gpu.h"
#include "cuda_runtime.h"

int main() {
    const int nx = 9;
    const int ny = 10;
    const int nz = 11;

    bindgen_handle *in_handle = make_data_store(nx, ny, nz);
    bindgen_handle *out_handle = make_data_store(nx, ny, nz);

    // Note that the order of the indices array here is k, j, i (which is the internal
    // Fortran layout). This is the layout that is expected by the bindings we have
    // written.
    float *in_array, *out_array;
    cudaMallocManaged((void **)&in_array, nx * ny * nz * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&out_array, nx * ny * nz * sizeof(float), cudaMemAttachGlobal);

    // fill some inputs
    float n1 = 0;
    float n2 = nz * ny * nx;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i, ++n1, --n2) {
                int index = k * ny * nx + j * nx + i;
                in_array[index] = n1;
                out_array[index] = n2;
            }

    // in the C bindings, the fortran array descriptors need to be filled explicitly
    bindgen_fortran_array_descriptor in_descriptor = {
        .rank = 3, .type = bindgen_fk_Float, .dims = {nx, ny, nz}, .data = in_array};
    bindgen_fortran_array_descriptor out_descriptor = {
        .rank = 3, .type = bindgen_fk_Float, .dims = {nx, ny, nz}, .data = out_array};

    transform_f_to_c(in_handle, &in_descriptor);
    run_copy_stencil(in_handle, out_handle);
    transform_c_to_f(&out_descriptor, out_handle);

    // now, the output can be verified
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = k * ny * nx + j * nx + i;
                if (in_array[idx] != out_array[idx]) {
                    printf("Error at position (k=%i, j=%i, i=%i)\n", k, j, i);
                    return 1;
                }
            }

    printf("It works!\n");

    cudaFree(in_array);
    cudaFree(out_array);

    bindgen_release(in_handle);
    bindgen_release(out_handle);
}
