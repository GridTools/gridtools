/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @brief provides a wrapper to execute a pure function on device returning a boolean
 * The function has to be passed as a functor with a static apply() method.
 */

#pragma once

#include <gridtools/common/cuda_util.hpp>

template <typename F, typename... Types>
__global__ void test_kernel(bool *result, Types... types) {
    *result = F::apply(types...);
}

template <typename F, typename... Types>
bool cuda_test(Types... types) {
    bool *d_result;
    GT_CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));
    test_kernel<F><<<1, 1>>>(d_result, types...);
    bool h_result;
    GT_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    return h_result;
}
