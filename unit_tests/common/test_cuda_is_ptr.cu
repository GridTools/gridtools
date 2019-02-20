/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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
