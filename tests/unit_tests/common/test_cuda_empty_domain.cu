/*
 * GridTools
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/cuda_util.hpp>

#include <cuda_test_helper.hpp>
#include <gtest/gtest.h>

__global__ void kernel() {}

TEST(cuda_launch, empty_kernel) {
    dim3 blocks{0, 0, 0};
    dim3 threads{8, 8, 8};
    cudaStream_t stream = nullptr;
    gridtools::cuda_util::launch(blocks, threads, 0, stream, kernel);
}
