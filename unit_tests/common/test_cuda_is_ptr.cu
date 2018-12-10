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

#include <gridtools/common/cuda_is_ptr.hpp>

#include <utility>

#include <gtest/gtest.h>

#include <gridtools/common/cuda_util.hpp>

using gridtools::is_gpu_ptr;
using gridtools::cuda_util::cuda_malloc;

TEST(test_is_gpu_ptr, host_ptr_is_no_cuda_ptr) {
    auto testee = std::unique_ptr<double>(new double);
    EXPECT_FALSE(is_gpu_ptr(testee.get()));
    EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

TEST(test_is_gpu_ptr, cuda_ptr_is_cuda_ptr) {
    auto testee = cuda_malloc<double>();
    EXPECT_TRUE(is_gpu_ptr(testee.get()));
    EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

TEST(test_is_gpu_ptr, cudaMallocHost_ptr_is_not_cuda_ptr) {
    auto testee = std::shared_ptr<double>(
        []() {
            double *res;
            GT_CUDA_CHECK(cudaMallocHost(&res, sizeof(double)));
            return res;
        }(),
        cudaFreeHost);
    EXPECT_FALSE(is_gpu_ptr(testee.get()));
    EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

TEST(test_is_gpu_ptr, cuda_ptr_inner_region_are_cuda_ptr) {
    auto testee = cuda_malloc<double>(2);
    EXPECT_TRUE(is_gpu_ptr(testee.get() + 1));
    EXPECT_EQ(cudaSuccess, cudaGetLastError());
}
