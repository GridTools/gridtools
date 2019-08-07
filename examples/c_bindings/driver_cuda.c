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

#include "copy_stencil_lib_cuda.h"
#include "cuda_runtime.h"

int main() {
    const int nx = 9;
    const int ny = 10;
    const int nz = 11;

    bindgen_handle *grid_handle = make_grid(nx, ny, nz);
    bindgen_handle *storage_info_handle = make_storage_info(nx, ny, nz);
    bindgen_handle *in_handle = make_data_store(storage_info_handle);
    bindgen_handle *out_handle = make_data_store(storage_info_handle);
    bindgen_handle *computation_handle = make_copy_stencil(grid_handle);
    free(storage_info_handle);
    free(grid_handle);

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
        .rank = 3, .type = bindgen_fk_Float, .dims = {nx, ny, nz}, .data = (void *)in_array};
    bindgen_fortran_array_descriptor out_descriptor = {
        .rank = 3, .type = bindgen_fk_Float, .dims = {nx, ny, nz}, .data = (void *)out_array};

    transform_f_to_c(in_handle, &in_descriptor);
    run_stencil(computation_handle, in_handle, out_handle);
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

    free(in_handle);
    free(out_handle);
    free(computation_handle);

    return 0;
}
