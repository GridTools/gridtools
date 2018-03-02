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
// using result_t = gridtools::array< size_t, Size * Size >;

GT_FUNCTION int linear_index(gridtools::array< size_t, 2 > &index) { return index[0] * Size + index[1]; }

__global__ void test_kernel1234(int *out_ptr) {
    //    int &out = *out_ptr;
    //    printf("bla\n");

    //    for (size_t i = 0; i < Size * Size; ++i)
    //        out[i] = 0;
    //    for (size_t i = 0; i < Size * Size; ++i)
    //        out_ptr[i] = 1;
    //
    //    auto cube_view = make_hypercube_view(hypercube< 2 >{range{0, Size}, range{0, Size}});
    //
    //    // fill the array with its linearized index
    //    for (auto pos : cube_view) {
    //        printf("pos: %d/%d\n", pos[0], pos[1]);
    //        out_ptr[linear_index(pos)] = linear_index(pos);
    //    }
    out_ptr[0] = 123;
    out_ptr[2] = 123;
};

TEST(multi_iterator, iterate_on_device) {
    int *out;
    cudaMalloc(&out, sizeof(int) * Size * Size);

    //        test_kernel1234<<< 32, 32 >>>(out);
    //    cudaDeviceSynchronize();

    int host_out[Size * Size];
    cudaMemcpy(&host_out, out, sizeof(int) * Size * Size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < Size * Size; ++i)
        ASSERT_EQ(i, host_out[i]) << "at i = " << i;
}
