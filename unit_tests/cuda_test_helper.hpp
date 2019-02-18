/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @brief provides a wrapper to execute a pure function on device returning a boolean
 * The function has to be passed as a functor with a static apply() method.
 */

#pragma once

template <typename F, typename... Types>
__global__ void test_kernel(bool *result, Types... types) {
    *result = F::apply(types...);
}

template <typename F, typename... Types>
bool cuda_test(Types... types) {
    bool *d_result;
    cudaMalloc(&d_result, sizeof(bool));
    test_kernel<F><<<1, 1>>>(d_result, types...);
    bool h_result;
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    return h_result;
}
