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

int get_index(int *strides, int i, int j, int k) { return i * strides[0] + j * strides[1] + k * strides[2]; }

void make_array_info(int *dims, int *strides, int *size, int Ni, int Nj, int Nk) {
#ifdef C_INTERFACE_EXAMPLE_PADDING // TODO should be moved to a regression test
    int pad = 1;
#else
    int pad = 0;
#endif
    dims[2] = Nk;
    strides[2] = 1 + pad;
    dims[1] = Nj;
    strides[1] = dims[2] * strides[2];
    dims[0] = Ni;
    strides[0] = dims[1] * strides[1];

    *size = get_index(strides, Ni - 1, Nj - 1, Nk - 1) + 1;
}

void fill_array(int *dims, int *strides, float *array, float value) {
    for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
            for (int k = 0; k < dims[2]; ++k) {
                array[get_index(strides, i, j, k)] = value;
            }
}

void fill_array_unique(int *dims, int *strides, float *array) {
    for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
            for (int k = 0; k < dims[2]; ++k) {
                array[get_index(strides, i, j, k)] = i * 100 + j * 10 + k;
            }
}

bool verify(int *dims, int *strides, float *expected, float *actual) {
    for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
            for (int k = 0; k < dims[2]; ++k) {
                if (expected[get_index(strides, i, j, k)] != actual[get_index(strides, i, j, k)]) {
                    printf("expected: %f, actual: %f for index %d/%d/%d\n",
                        expected[get_index(strides, i, j, k)],
                        actual[get_index(strides, i, j, k)],
                        i,
                        j,
                        k);
                    return false;
                }
            }
    return true;
}
