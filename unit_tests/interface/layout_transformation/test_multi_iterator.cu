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

#include "test_multi_iterator.cpp"

static const uint_t Size = 2;
using result_t = gridtools::array< size_t, Size * Size >;

GT_FUNCTION int linear_index(size_t a, size_t b) { return a * Size + b; }

__global__ void exec(result_t *out_ptr) {
    gridtools::array< uint_t, 2 > dims{Size, Size};
    result_t &out = *out_ptr;

    for (size_t i = 0; i < Size * Size; ++i)
        out[i] = 0;

    // fill the array with its linearized index
    iterate(dims, [&](size_t a, size_t b) { out[linear_index(a, b)] = linear_index(a, b); });
};

TEST(multi_iterator, iterate_on_device) {
    result_t *out;
    cudaMalloc(&out, sizeof(result_t));

    exec<<< 1, 1 >>>(out);

    result_t host_out;
    cudaMemcpy(&host_out, out, sizeof(result_t), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < Size * Size; ++i)
        ASSERT_EQ(i, host_out[i]);
}
