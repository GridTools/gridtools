/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include <stdio.h>
#include <stdlib.h>

#include "implementation.h"

#if FLOAT_PRECISION == 4
typedef float float_type;
#elif FLOAT_PRECISION == 8
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
    gt_handle *in_handle, *out_handle, *stencil;

    init_in(in);

    in_handle = create_data_store(I, J, K, (float_type *)in);
    out_handle = create_data_store(I, J, K, (float_type *)out);
    stencil = create_copy_stencil(in_handle, out_handle);

    steady_stencil(stencil);
    run_stencil(stencil);
    sync_data_store(in_handle);
    sync_data_store(out_handle);

    verify("in", in);
    verify("out", out);

    gt_release(stencil);
    gt_release(in_handle);
    gt_release(out_handle);

    printf("It works!\n");
}
