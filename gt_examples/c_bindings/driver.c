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

#include "copy_stencil.h"

int main() {
    gt_handle *wrapper_handle = make_wrapper(9, 10, 11);
    gt_handle *computation_handle = make_copy_stencil(wrapper_handle);

    // Note that the order of the indices array here is k, j, i (which is the internal
    // Fortran layout). This is the layout that is expected by the bindings we have
    // written.
    float in_array[11][10][9];
    float out_array[11][10][9];

    // fill some inputs
    float n1 = 0;
    float n2 = 11 * 10 * 9;
    for (int k = 0; k < 11; ++k)
        for (int j = 0; j < 10; ++j)
            for (int i = 0; i < 9; ++i, ++n1, --n2) {
                in_array[k][j][i] = n1;
                out_array[k][j][i] = n2;
            }

    // in the C bindings, the fortran array descriptors need to be filled explicitly
    gt_fortran_array_descriptor in_descriptor = {
        .rank = 3, .type = gt_fk_Float, .dims = {9, 10, 11}, .data = (void *)in_array};
    gt_fortran_array_descriptor out_descriptor = {
        .rank = 3, .type = gt_fk_Float, .dims = {9, 10, 11}, .data = (void *)out_array};

    run_stencil(wrapper_handle, computation_handle, &in_descriptor, &out_descriptor);

    // now, the output can be verified
    for (int k = 1; k < 11; ++k)
        for (int j = 0; j < 10; ++j)
            for (int i = 0; i < 9; ++i)
                if (in_array[k][j][i] != out_array[k][j][i]) {
                    printf("Error at position (k=%i, j=%i, i=%i)\n", k, j, i);
                    return 1;
                }

    printf("It works!\n");
    return 0;
}

