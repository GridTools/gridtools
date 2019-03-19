/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "test_accumulate.hpp"
#include "gtest/gtest.h"
#include <gridtools/common/cuda_util.hpp>

__global__ void accumulate_and_kernel(bool *result) { *result = test_accumulate_and(); }

__global__ void accumulate_or_kernel(bool *result) { *result = test_accumulate_or(); }

TEST(accumulate, test_and) {
    bool result;
    bool *resultDevice;
    GT_CUDA_CHECK(cudaMalloc(&resultDevice, sizeof(bool)));

    accumulate_and_kernel<<<1, 1>>>(resultDevice);

    GT_CUDA_CHECK(cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(result);
}

TEST(accumulate, test_or) {
    bool result;
    bool *resultDevice;
    GT_CUDA_CHECK(cudaMalloc(&resultDevice, sizeof(bool)));

    accumulate_or_kernel<<<1, 1>>>(resultDevice);

    GT_CUDA_CHECK(cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(result);
}
