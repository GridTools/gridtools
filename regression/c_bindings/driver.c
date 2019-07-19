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

#if GT_FLOAT_PRECISION == 4
#include "implementation_float.h"
typedef float float_type;
#elif GT_FLOAT_PRECISION == 8
#include "implementation_double.h"
typedef double float_type;
#else
#error float precision not properly set (4 or 8 bytes supported)
#endif

#define I 9
#define J 10
#define K 11

float_type initial_value(int i, int j, int k) { return i + j + k; }

void init_in(float_type arr[I][J][K]) {
    int i, j, k;
    for (i = 0; i != I; ++i)
        for (j = 0; j != J; ++j)
            for (k = 0; k != K; ++k)
                arr[i][j][k] = initial_value(i, j, k);
}

void verify(const char *label, float_type arr[I][J][K]) {
    int i, j, k;
    for (i = 0; i != I; ++i)
        for (j = 0; j != J; ++j)
            for (k = 0; k != K; ++k)
                if (arr[i][j][k] != initial_value(i, j, k)) {
                    fprintf(stderr,
                        "data mismatch in %s[%d][%d][%d]: actual - %f , expected - %f\n",
                        label,
                        i,
                        j,
                        k,
                        arr[i][j][k],
                        initial_value(i, j, k));
                    exit(i);
                }
}

int main() {
    float_type in[I][J][K];
    float_type out[I][J][K];
    bindgen_handle *in_handle, *out_handle, *stencil;

    init_in(in);

    in_handle = create_data_store(I, J, K, (float_type *)in);
    out_handle = create_data_store(I, J, K, (float_type *)out);
    stencil = create_copy_stencil(in_handle, out_handle);

    run_stencil(stencil);
    sync_data_store(in_handle);
    sync_data_store(out_handle);

    verify("in", in);
    verify("out", out);

    bindgen_release(stencil);
    bindgen_release(in_handle);
    bindgen_release(out_handle);

    printf("It works!\n");
}
